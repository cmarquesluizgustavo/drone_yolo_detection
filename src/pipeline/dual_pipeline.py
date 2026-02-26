import cv2
import os
from pathlib import Path

from pipeline.single_pipeline import DetectionPipeline
from pipeline.detection_fusion import DualDroneFusion, Detection
from camera import Camera
from stats import DetectionStatistics
from position_estimation import (
    distance_from_geoposition,
    fuse_avg_position_from_distance_bearing,
    fuse_avg_position_from_distance_bearing_weighted,
    fuse_triangulate_position_from_bearing_intersection
)
import viewer
from plots import plot_ground_plane, build_series_from_frame
from geoconverter import GeoConverter

class DualDronePipeline:

    def __init__(
        self,
        model_path,
        person_confidence_threshold,
        enable_weapon_detection,
        weapon_confidence_threshold,
        sample_majority_threshold,
        association_threshold,
        device,
    ):

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

        self.fusion = DualDroneFusion(association_threshold_m=association_threshold, weapon_threshold=weapon_confidence_threshold)

        self.camera_drone1 = Camera()
        self.camera_drone2 = Camera()

        # Keep separate stats so we can compare single-drone vs fusion quality.
        self.stats_drone1 = self.pipeline_drone1.stats
        self.stats_drone2 = self.pipeline_drone2.stats
        self.stats_fused = DetectionStatistics(sample_majority_threshold=sample_majority_threshold)

        self.save_crops = True
        self.enable_weapon_detection = enable_weapon_detection
        self.person_confidence_threshold = person_confidence_threshold
        self.verbose = True  # set True to print per-frame debug output
        self.show_weapon_confidence = False

        # Optional: write a 2D ground-plane (local XY) plot per sample.
        self.enable_ground_plane_plot = False

        self.ground_plane_plot_filename = "ground_plane.png"

    def per_frame_output(
        self,
        det1,
        det2,
        fused_detections,
        w1_conf,
        w2_conf,
        w_fused
    ):
        best_fused_avg = next((d for d in fused_detections if d.get("source") == "fused_average"), None)
        best_fused_bi = next((d for d in fused_detections if d.get("source") == "fused_bearing_intersection"), None)
        best_fused_weighted_avg = next((d for d in fused_detections if d.get("source") == "fused_weighted_average"), None)
        # bets_fused_weighted_bi = next((d for d in fused_detections if d.get("source") == "fused_weighted_bearing_intersection"), None)

        d1_h = det1.get("distance_pinhole_m") if det1 else None
        d1_t = det1.get("distance_pitch_m") if det1 else None
        #d1_n = det1.get("distance_new_m") if det1 else None
        d2_h = det2.get("distance_pinhole_m") if det2 else None
        d2_t = det2.get("distance_pitch_m") if det2 else None
        #d2_n = det2.get("distance_new_m") if det2 else None

        geo1 = det1.get("person_geoposition") if det1 else None
        geo2 = det2.get("person_geoposition") if det2 else None

        # Average fusion method
        geo_fused_avg = best_fused_avg.get("fused_geoposition_average") if best_fused_avg else None
        d1_fused_avg = best_fused_avg.get("distance_drone1_average_m") if best_fused_avg else None
        d2_fused_avg = best_fused_avg.get("distance_drone2_average_m") if best_fused_avg else None

        # Bearing intersection fusion method
        geo_fused_bi = best_fused_bi.get("fused_geoposition_bearing_intersection") if best_fused_bi else None
        d1_fused_bi = best_fused_bi.get("distance_drone1_bearing_intersection_m") if best_fused_bi else None
        d2_fused_bi = best_fused_bi.get("distance_drone2_bearing_intersection_m") if best_fused_bi else None

        geo_weighted_fused_avg = best_fused_weighted_avg.get("fused_geoposition_weighted_average") if best_fused_weighted_avg else None
        d1_weighted_fused_avg = best_fused_weighted_avg.get("distance_drone1_weighted_average_m") if best_fused_weighted_avg else None
        d2_weighted_fused_avg = best_fused_weighted_avg.get("distance_drone2_weighted_average_m") if best_fused_weighted_avg else None
        
        # geo_weighted_fused_bi = bets_fused_weighted_bi.get("fused_geoposition_weighted_bearing_intersection") if bets_fused_weighted_bi else None
        # d1_weighted_fused_bi = bets_fused_weighted_bi.get("distance_drone1_weighted_bearing_intersection_m") if bets_fused_weighted_bi else None
        # d2_weighted_fused_bi = bets_fused_weighted_bi.get("distance_drone2_weighted_bearing_intersection_m") if bets_fused_weighted_bi else None


        def fmt(val):
            """Format numeric values safely. If val is a tuple/list, format its numeric
            elements joined by '/'. Fallback to str() for unexpected types.
            """
            if val is None:
                return "None"
            # If it's a tuple/list, try to format numeric elements
            if isinstance(val, (tuple, list)):
                try:
                    parts = []
                    for v in val:
                        parts.append(f"{float(v):.2f}")
                    return "(" + ",".join(parts) + ")"
                except Exception:
                    return str(val)
            # Single numeric value
            try:
                return f"{float(val):.2f}"
            except Exception:
                return str(val)

        def fmt_geo(geo):
            if geo and isinstance(geo, dict):
                return f"({geo.get('latitude', 0):.6f},{geo.get('longitude', 0):.6f})"
            return "None"

        print(
            "DISTANCE: "
            f"dist1_height-based={fmt(d1_h)}, "
            f"dist1_pitch-based={fmt(d1_t)}, "
            #f"dist1_new={fmt(d1_n)}, "
            f"dist2_height-based={fmt(d2_h)}, "
            f"dist2_pitch-based={fmt(d2_t)}, "
            #f"dist2_new={fmt(d2_n)}, "
            f"geoposition_drone1={fmt_geo(geo1)}, " # this geopositions use only the pitch-based.
            f"geoposition_drone2={fmt_geo(geo2)}"
        )
        print(
            "FUSED_AVERAGE: "
            f"dist1_from_fused_avg={fmt(d1_fused_avg)}, "
            f"dist2_from_fused_avg={fmt(d2_fused_avg)}, "
            f"geoposition_fused_avg={fmt_geo(geo_fused_avg)}"
        )
        print(
            "FUSED_BEARING_INTERSECTION: "
            f"dist1_from_fused_bi={fmt(d1_fused_bi)}, "
            f"dist2_from_fused_bi={fmt(d2_fused_bi)}, "
            f"geoposition_fused_bi={fmt_geo(geo_fused_bi)}"
        )
        print(
            "FUSED_WEIGHTED_AVERAGE: "
            f"dist1_from_weighted_fused_avg={fmt(d1_weighted_fused_avg)}, "
            f"dist2_from_weighted_fused_avg={fmt(d2_weighted_fused_avg)}, "
            f"geoposition_fused_weighted_avg={fmt_geo(geo_weighted_fused_avg)}"
        )
        # print(
        #     "FUSED_WEIGHTED_BEARING_INTERSECTION: "
        #     f"dist1_from_weighted_fused_bi={fmt(d1_weighted_fused_bi)}, "
        #     f"dist2_from_weighted_fused_bi={fmt(d2_weighted_fused_bi)}, "
        #     f"geoposition_fused_weighted_bi={fmt_geo(geo_weighted_fused_bi)}"
        # )
        print(
            "DETECTION: "
            f"w1={fmt(w1_conf)}, "
            f"w2={fmt(w2_conf)}, "
            f"w_fused={fmt(w_fused)}" # using the fusion formula for now
        )

    def process_all_samples(self, input_dir_drone1, input_dir_drone2, output_base_dir, only_angle=None):
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
                self.process_angle_batch(angle_dir_drone1, angle_dir_drone2, angle_output_dir, angle)

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

        else:
            # No angle subdirectories, process directly
            self.process_angle_batch(input_dir_drone1, input_dir_drone2, output_base_dir, None)
            for s in (self.stats_drone1, self.stats_drone2, self.stats_fused):
                s.finalize()

    def process_angle_batch(self, input_dir_drone1, input_dir_drone2, output_base_dir, angle=None):
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

            print(f"  Sample type: {sample_class.upper()} (has_weapon={sample_ground_truth})")

            # Mark start of new sample
            for s in (self.stats_drone1, self.stats_drone2, self.stats_fused):
                s.start_new_sample(sample_ground_truth, sample_class)

            # Process synchronized sample pair
            self.process_sample_frames(
                sample_path_drone1,
                sample_path_drone2,
                sample_name,
                detections_base_dir,
                crops_base_dir,
                fused_base_dir,
            )

        # Finalize statistics for this angle
        for s in (self.stats_drone1, self.stats_drone2, self.stats_fused):
            s.finalize()

    def process_sample_frames(
        self,
        sample_dir_drone1,
        sample_dir_drone2,
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
                    dist = frame_result.get("distances", {})

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

                            avg_list = gps.get("targets_fused_average") or []
                            if len(avg_list) > 0:
                                try:
                                    fused_lat = float(avg_list[-1][0])
                                    fused_lon = float(avg_list[-1][1])
                                except Exception:
                                    fused_lat, fused_lon = None, None

                            # Otherwise fall back to fused average
                            if fused_lat is None or fused_lon is None:
                                bi_list = gps.get("targets_fused_bearing_intersection") or []
                                if len(bi_list) > 0:
                                    try:
                                        fused_lat = float(bi_list[-1][0])
                                        fused_lon = float(bi_list[-1][1])
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
                                targets_1=gps.get("targets_d1"),
                                targets_2=gps.get("targets_d2"),
                                targets_fused_avg=gps.get("targets_fused_average"),
                                targets_fused_bi=gps.get("targets_fused_bearing_intersection"),
                                targets_fused_wavg=gps.get("targets_fused_weighted_average"),
                                #targets_fused_wbi=gps.get("targets_fused_weighted_bearing_intersection"),
                                measurements_1=gps.get("measurements_d1"),
                                measurements_2=gps.get("measurements_d2"),
                                distance_info=dist,
                                ticks_lim=tick_half_range,
                                out_path=out_path,
                                draw_bearing_rays=True,
                                draw_distance_circles=False
                            )

                            # print(f"[plot_ground_plane] Saved plot to {out_path}")
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

        # Ensure detector uses the correct camera object with loaded telemetry
        self.pipeline_drone1.detector.camera = self.camera_drone1
        self.pipeline_drone2.detector.camera = self.camera_drone2

        # # Debug: Print drone positions on first frame
        # if frame_idx == 0:
        #     print("  DEBUG Drone positions:")
        #     print(
        #         f"    Drone 1: lat={self.camera_drone1.lat:.6f}, "
        #         f"lon={self.camera_drone1.lon:.6f}, yaw={self.camera_drone1.yaw_deg:.1f}°"
        #     )
        #     print(
        #         f"    Drone 2: lat={self.camera_drone2.lat:.6f}, "
        #         f"lon={self.camera_drone2.lon:.6f}, yaw={self.camera_drone2.yaw_deg:.1f}°"
        #     )

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
        # Use PeopleDetector's detect_people_with_estimation method directly
        detections1 = self.pipeline_drone1.detector.detect_people_with_estimation(frame1_path, img1)
        detections2 = self.pipeline_drone2.detector.detect_people_with_estimation(frame2_path, img2)

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

        if detections1 and weapon_results1 and len(weapon_results1) == len(detections1):
            for d, wr in zip(detections1, weapon_results1):
                weapon_dets = wr.get("weapon_detections", []) or []
                w_conf = max((wd.get("confidence", 0.0) for wd in weapon_dets), default=0.0)
                d["weapon_confidence"] = float(w_conf)
                d["has_weapon"] = bool(wr.get("has_weapon", False))

        if detections2 and weapon_results2 and len(weapon_results2) == len(detections2):
            for d, wr in zip(detections2, weapon_results2):
                weapon_dets = wr.get("weapon_detections", []) or []
                w_conf = max((wd.get("confidence", 0.0) for wd in weapon_dets), default=0.0)
                d["weapon_confidence"] = float(w_conf)
                d["has_weapon"] = bool(wr.get("has_weapon", False))

        fused_detections = []

        # Convert dict detections -> Detection dataclass expected by detection_fusion.py
        def to_det(d, drone_id, frame_id):
            x = d.get("x")
            y = d.get("y")
            if x is None or y is None:
                lat = (d.get("lat") or (d.get("person_geoposition") or {}).get("latitude"))
                lon = (d.get("lon") or (d.get("person_geoposition") or {}).get("longitude"))
                if lat is not None and lon is not None:
                    x, y = GeoConverter.geo_to_xy(lat, lon)

            lat = d.get("lat") or (d.get("person_geoposition") or {}).get("latitude")
            lon = d.get("lon") or (d.get("person_geoposition") or {}).get("longitude")

            return Detection(
                bbox=tuple(d.get("bbox")) if d.get("bbox") is not None else (0,0,0,0),
                person_confidence=float(d.get("person_confidence", d.get("confidence", 0.0)) or 0.0),
                distance_m=float(d.get("distance_m") or 0.0),
                bearing_deg=float(d.get("bearing_deg", d.get("bearing", 0.0)) or 0.0),
                x=float(x) if x is not None else float("nan"),
                y=float(y) if y is not None else float("nan"),
                lat=float(lat) if lat is not None else 0.0,
                lon=float(lon) if lon is not None else 0.0,
                drone_id=int(drone_id),
                frame_id=int(frame_id),
                weapon_confidence=float(d.get("weapon_confidence", 0.0) or 0.0),
                has_weapon=bool(d.get("has_weapon", False)),
            )

        dets1_obj = [to_det(d, 1, frame_idx) for d in (detections1 or [])]
        dets2_obj = [to_det(d, 2, frame_idx) for d in (detections2 or [])]

        measurement_groups = self.fusion.matching(dets1_obj, dets2_obj, self.camera_drone1, self.camera_drone2)

        # print(f"[DEBUG] measurement_groups count: {len(measurement_groups)}")
        # for idx, g in enumerate(measurement_groups):
        #     ms = g.get("drone_measurements", [])
        #     print(f"[DEBUG] measurement_group[{idx}] drone_measurements count: {len(ms)}")

        for g in measurement_groups:
            ms = g.get("drone_measurements", [])
            if len(ms) != 2:
                continue  # only do dual-drone geo fusion on paired groups

            # recover the original per-drone bboxes for overlay matching
            bbox1 = g.get("bbox_drone1")
            bbox2 = g.get("bbox_drone2")

            print(f"BBOX1: {bbox1}, BBOX2: {bbox2}")

            d1 = ms[0]
            d2 = ms[1]

            print(f"conf1: {d1['person_confidence']}, dist1: {d1['distance']}, bearing1: {d1['bearing']}")
            print(f"conf2: {d2['person_confidence']}, dist2: {d2['distance']}, bearing2: {d2['bearing']}")

            avg_latlon = fuse_avg_position_from_distance_bearing(
                self.camera_drone1, d1["distance"], d1["bearing"],
                self.camera_drone2, d2["distance"], d2["bearing"],
            )
            #print(f"[DEBUG] avg_latlon: {avg_latlon}")

            tri_latlon = fuse_triangulate_position_from_bearing_intersection(
                self.camera_drone1, d1["bearing"],
                self.camera_drone2, d2["bearing"],
            )
            #print(f"[DEBUG] tri_latlon: {tri_latlon}")

            # IMPORTANT: weapon/person confidence & has_weapon come from detection_fusion.py
            fused_person_conf = g.get("person_confidence", 0.0)
            fused_weapon_conf = g.get("weapon_confidence", 0.0)
            fused_has_weapon = g.get("has_weapon", False)

            weighted_avg_latlon = fuse_avg_position_from_distance_bearing_weighted(
                self.camera_drone1, d1["distance"], d1["bearing"], bbox1, d1["person_confidence"],
                self.camera_drone2, d2["distance"], d2["bearing"], bbox2, d2["person_confidence"]
            )
            #print(f"[DEBUG] weighted_avg_latlon: {weighted_avg_latlon}")

            # weighted_tri_latlon = fuse_triangulate_position_from_bearing_intersection_weighted(
            #     self.camera_drone1, d1["distance"], d1["bearing"], bbox1, d1["person_confidence"],
            #     self.camera_drone2, d2["distance"], d2["bearing"], bbox2, d2["person_confidence"]
            # )

            # Average method result
            avg_dist1 = distance_from_geoposition(self.camera_drone1, avg_latlon[0], avg_latlon[1])
            avg_dist2 = distance_from_geoposition(self.camera_drone2, avg_latlon[0], avg_latlon[1])
            fused_detections.append({
                "bbox_drone1": bbox1,
                "bbox_drone2": bbox2,
                "person_confidence": fused_person_conf,
                "has_weapon": fused_has_weapon,
                "weapon_confidence": float(fused_weapon_conf),
                "fused_geoposition_average": {"latitude": avg_latlon[0], "longitude": avg_latlon[1]},
                "distance_drone1_average_m": avg_dist1,
                "distance_drone2_average_m": avg_dist2,
                "source": "fused_average",
            })

            # Bearing intersection method result
            tri_dist1 = distance_from_geoposition(self.camera_drone1, tri_latlon[0], tri_latlon[1])
            tri_dist2 = distance_from_geoposition(self.camera_drone2, tri_latlon[0], tri_latlon[1])
            fused_detections.append({
                "bbox_drone1": bbox1,
                "bbox_drone2": bbox2,
                "person_confidence": fused_person_conf,
                "has_weapon": fused_has_weapon,
                "weapon_confidence": float(fused_weapon_conf),
                "fused_geoposition_bearing_intersection": {"latitude": tri_latlon[0], "longitude": tri_latlon[1]},
                "distance_drone1_bearing_intersection_m": tri_dist1,
                "distance_drone2_bearing_intersection_m": tri_dist2,
                "source": "fused_bearing_intersection",
            })

            weighted_avg_dist1 = distance_from_geoposition(self.camera_drone1, weighted_avg_latlon[0], weighted_avg_latlon[1])
            weighted_avg_dist2 = distance_from_geoposition(self.camera_drone2, weighted_avg_latlon[0], weighted_avg_latlon[1])

            fused_detections.append({
                "bbox_drone1": bbox1,
                "bbox_drone2": bbox2,
                "person_confidence": fused_person_conf,
                "has_weapon": fused_has_weapon,
                "weapon_confidence": float(fused_weapon_conf),

                "fused_geoposition_weighted_average": {
                    "latitude": weighted_avg_latlon[0],
                    "longitude": weighted_avg_latlon[1]
                },

                "distance_drone1_weighted_average_m": weighted_avg_dist1,
                "distance_drone2_weighted_average_m": weighted_avg_dist2,

                "source": "fused_weighted_average"
            })

            # weighted_tri_dist1 = distance_from_geoposition(self.camera_drone1, weighted_tri_latlon[0], weighted_tri_latlon[1])
            # weighted_tri_dist2 = distance_from_geoposition(self.camera_drone2, weighted_tri_latlon[0], weighted_tri_latlon[1])

            # fused_detections.append({
            #     "bbox_drone1": bbox1,
            #     "bbox_drone2": bbox2,
            #     "person_confidence": fused_person_conf,
            #     "has_weapon": fused_has_weapon,
            #     "weapon_confidence": float(fused_weapon_conf),

            #     "fused_geoposition_weighted_bearing_intersection": {
            #         "latitude": weighted_tri_latlon[0],
            #         "longitude": weighted_tri_latlon[1]
            #     },

            #     "distance_drone1_weighted_bearing_intersection_m": weighted_tri_dist1,
            #     "distance_drone2_weighted_bearing_intersection_m": weighted_tri_dist2,

            #     "source": "fused_weighted_bearing_intersection"
            # })



        # Extract fused pairs for distance estimates
        fused_avg_pairs_d1 = [
            (d["distance_drone1_average_m"], d["distance_drone2_average_m"])
            for d in fused_detections if d.get("source") == "fused_average"
        ]
        fused_avg_pairs_d2 = [
            (d["distance_drone2_average_m"], d["distance_drone1_average_m"])
            for d in fused_detections if d.get("source") == "fused_average"
        ]
        fused_bi_pairs_d1 = [
            (d["distance_drone1_bearing_intersection_m"], d["distance_drone2_bearing_intersection_m"])
            for d in fused_detections if d.get("source") == "fused_bearing_intersection"
        ]
        fused_bi_pairs_d2 = [
            (d["distance_drone2_bearing_intersection_m"], d["distance_drone1_bearing_intersection_m"])
            for d in fused_detections if d.get("source") == "fused_bearing_intersection"
        ]
        fused_weighted_avg_d1 = [
            (d["distance_drone1_weighted_average_m"], d["distance_drone2_weighted_average_m"])
            for d in fused_detections if d.get("source") == "fused_weighted_average"
        ]
        fused_weighted_avg_d2 = [
            (d["distance_drone2_weighted_average_m"], d["distance_drone1_weighted_average_m"])
            for d in fused_detections if d.get("source") == "fused_weighted_average"
        ]
        # fused_weighted_tri_d1 = [
        #     (d["distance_drone1_weighted_bearing_intersection_m"], d["distance_drone2_weighted_bearing_intersection_m"])
        #     for d in fused_detections if d.get("source") == "fused_weighted_bearing_intersection"
        # ]
        # fused_weighted_tri_d2 = [
        #     (d["distance_drone2_weighted_bearing_intersection_m"], d["distance_drone1_weighted_bearing_intersection_m"])
        #     for d in fused_detections if d.get("source") == "fused_weighted_bearing_intersection"
        # ]


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
                    if isinstance(d, dict) and d.get("source") in ("fused_average", "fused_bearing_intersection", "fused_weighted_average")#, "fused_weighted_bearing_intersection")
                ],
            )
        except Exception:
            frame_series = None

        # Verbose output for fusion (fixed source labels)
        if self.verbose or (frame_idx % 10 == 0):
            fused_count = len(
                [d for d in fused_detections if d.get("source") in ("fused_average", "fused_bearing_intersection", "fused_weighted_average")]#, "fused_weighted_bearing_intersection")]
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
            has_weapon_gt = (sample_meta.get("sample_class") == "real")
        else:
            has_weapon_gt = sample_name.lower().startswith("real")

        #weapons_fused = sum(1 for d in fused_detections if d.get("has_weapon", False))

        sample_class = (
            sample_meta.get("sample_class")
            if (sample_meta or {}).get("sample_class") in ("real", "falso")
            else ("real" if has_weapon_gt else "falso")
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
        #pairs1_new = []
        #pairs2_new = []

        if real_distance is not None:
            if best_det1:
                if best_det1.get("distance_m") is not None:
                    pairs1_primary.append((best_det1["distance_m"], real_distance))
                if best_det1.get("distance_pinhole_m") is not None:
                    pairs1_p.append((best_det1["distance_pinhole_m"], real_distance))
                if best_det1.get("distance_pitch_m") is not None:
                    pairs1_pitch.append((best_det1["distance_pitch_m"], real_distance))
                # if best_det1.get("distance_new_m") is not None:
                #     pairs1_new.append((best_det1["distance_new_m"], real_distance))
            if best_det2:
                if best_det2.get("distance_m") is not None:
                    pairs2_primary.append((best_det2["distance_m"], real_distance))
                if best_det2.get("distance_pinhole_m") is not None:
                    pairs2_p.append((best_det2["distance_pinhole_m"], real_distance))
                if best_det2.get("distance_pitch_m") is not None:
                    pairs2_pitch.append((best_det2["distance_pitch_m"], real_distance))
                # if best_det2.get("distance_new_m") is not None:
                #     pairs2_new.append((best_det2["distance_new_m"], real_distance))

        # Weapon detection for best detection only
        weapons_detected_d1_best = 0
        people_with_weapons_d1 = 0
        w1_conf = 0.0
        if best_det1 and weapon_results1:
            best_idx1 = detections1.index(best_det1)
            if best_idx1 < len(weapon_results1):
                wr1 = weapon_results1[best_idx1]
                if wr1.get("has_weapon", False):
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
                if wr2.get("has_weapon", False):
                    weapons_detected_d2_best = len(wr2.get("weapon_detections", []))
                    people_with_weapons_d2 = 1
                wdets2 = wr2.get("weapon_detections", [])
                if wdets2:
                    w2_conf = max(w.get("weapon_confidence", w.get("confidence", 0.0)) for w in wdets2)

        # Permitir comparação de diferentes métodos de fuse_confidence
        if getattr(self, "verbose", False):
            w_fused_1 = self.fusion.fuse_confidence(w1_conf, w2_conf)
            self.per_frame_output(best_det1, best_det2, fused_detections, w1_conf, w2_conf, w_fused_1)

        # Individual drone stats (using only best detection)
        # Construct distance_estimates for Drone 1
        distance_estimates_d1 = []
        if pairs1_p:
            distance_estimates_d1.extend({"est": p[0], "method": "pinhole"} for p in pairs1_p)
        if pairs1_pitch:
            distance_estimates_d1.extend({"est": p[0], "method": "pitch"} for p in pairs1_pitch)
        # if pairs1_new:
        #     distance_estimates_d1.extend({"est": p[0], "method": "new"} for p in pairs1_new)

        self.stats_drone1.add_image_results(
            1 if best_det1 else 0,
            weapons_detected_d1_best,
            people_with_weapons_d1,
            has_weapon_gt,
            distances=distances1,
            real_distance=real_distance,
            cam_height_m=cam_height_m,
            sample_class=sample_class,
            distance_estimates=distance_estimates_d1,
        )

        # Construct distance_estimates for Drone 2
        distance_estimates_d2 = []
        if pairs2_p:
            distance_estimates_d2.extend({"est": p[0], "method": "pinhole"} for p in pairs2_p)
        if pairs2_pitch:
            distance_estimates_d2.extend({"est": p[0], "method": "pitch"} for p in pairs2_pitch)
        # if pairs2_new:
        #     distance_estimates_d2.extend({"est": p[0], "method": "new"} for p in pairs2_new)

        self.stats_drone2.add_image_results(
            1 if best_det2 else 0,
            weapons_detected_d2_best,
            people_with_weapons_d2,
            has_weapon_gt,
            distances=distances2,
            real_distance=real_distance,
            cam_height_m=cam_height_m,
            sample_class=sample_class,
            distance_estimates=distance_estimates_d2,
        )

        # Construct distance_estimates for fused statistics
        distance_estimates_fused = []
        if fused_avg_pairs_d1:
            distance_estimates_fused.extend({"est": p[0], "method": "fused", "fusion_type": "avg", "d_source": "d1"} for p in fused_avg_pairs_d1)
        if fused_avg_pairs_d2:
            distance_estimates_fused.extend({"est": p[0], "method": "fused", "fusion_type": "avg", "d_source": "d2"} for p in fused_avg_pairs_d2)
        if fused_bi_pairs_d1:
            distance_estimates_fused.extend({"est": p[0], "method": "fused", "fusion_type": "bi", "d_source": "d1"} for p in fused_bi_pairs_d1)
        if fused_bi_pairs_d2:
            distance_estimates_fused.extend({"est": p[0], "method": "fused", "fusion_type": "bi", "d_source": "d2"} for p in fused_bi_pairs_d2)
        if fused_weighted_avg_d1:
            distance_estimates_fused.extend({"est": p[0], "method": "fused", "fusion_type": "weighted_avg", "d_source": "d1"} for p in fused_weighted_avg_d1)
        if fused_weighted_avg_d2:
            distance_estimates_fused.extend({"est": p[0], "method": "fused", "fusion_type": "weighted_avg", "d_source": "d2"} for p in fused_weighted_avg_d2)
        # if fused_weighted_tri_d1:
        #     distance_estimates_fused.extend({"est": p[0], "method": "fused", "fusion_type": "weighted_bi", "d_source": "d1"} for p in fused_weighted_tri_d1)
        # if fused_weighted_tri_d2:
        #     distance_estimates_fused.extend({"est": p[0], "method": "fused", "fusion_type": "weighted_bi", "d_source": "d2"} for p in fused_weighted_tri_d2)

        # ---- FUSED stats should be FRAME-LEVEL (one target max), not “two methods = two people” ----
        fused_present = any(
            isinstance(d, dict) and d.get("source") in ("fused_average", "fused_bearing_intersection", "fused_weighted_average")#, "fused_weighted_bearing_intersection")
            for d in (fused_detections or [])
        )

        fused_frame_has_weapon = any(
            isinstance(d, dict) and bool(d.get("has_weapon", False))
            for d in (fused_detections or [])
        )

        self.stats_fused.add_image_results(
            1 if fused_present else 0,                 # num_people (frame-level)
            1 if fused_frame_has_weapon else 0,        # num_weapons (frame-level boolean)
            1 if fused_frame_has_weapon else 0,        # people_with_weapons_count (frame-level)
            has_weapon_gt,                            # use same GT logic as UAV stats
            distances=None,                            # ok to keep None unless you want fused distance histograms
            real_distance=real_distance,
            cam_height_m=cam_height_m,
            sample_class=sample_class,                 # use the already-derived class
            distance_estimates=distance_estimates_fused # keep RMSE logging intact
        )


        def collect_distances(best_det1, best_det2, fused_detections):
            out = {}

            # --- Drone 1 ---
            if best_det1:
                out["d1_pitch"] = best_det1.get("distance_pitch_m")
                out["d1_pinhole"] = best_det1.get("distance_pinhole_m")
                # out["d1_new"] = best_det1.get("distance_new_m")

            # --- Drone 2 ---
            if best_det2:
                out["d2_pitch"] = best_det2.get("distance_pitch_m")
                out["d2_pinhole"] = best_det2.get("distance_pinhole_m")
                # out["d2_new"] = best_det2.get("distance_new_m")

            # --- Fused ---
            for d in fused_detections:
                if d.get("source") == "fused_average":
                    out["fused_avg_d1"] = d.get("distance_drone1_average_m")
                    out["fused_avg_d2"] = d.get("distance_drone2_average_m")

                elif d.get("source") == "fused_bearing_intersection":
                    out["fused_bi_d1"] = d.get("distance_drone1_bearing_intersection_m")
                    out["fused_bi_d2"] = d.get("distance_drone2_bearing_intersection_m")

                elif d.get("source") == "fused_weighted_average":
                    out["fused_weighted_avg_d1"] = d.get("distance_drone1_weighted_average_m")
                    out["fused_weighted_avg_d2"] = d.get("distance_drone2_weighted_average_m")

                # elif d.get("source") == "fused_weighted_bearing_intersection":
                #     out["fused_weighted_bi_d1"] = d.get("distance_drone1_weighted_bearing_intersection_m")
                #     out["fused_weighted_bi_d2"] = d.get("distance_drone2_weighted_bearing_intersection_m")

            return out
        
        distances = collect_distances(best_det1, best_det2, fused_detections)

        # -------------------------------
        # Frame winner analysis
        # -------------------------------
        if real_distance is not None:

            pinhole_vals = [v for k, v in distances.items() if "pinhole" in k and v is not None]
            pitch_vals   = [v for k, v in distances.items() if "pitch" in k and v is not None]
            # new_vals     = [v for k, v in distances.items() if "new" in k and v is not None]

            err_pinhole = sum(abs(v - real_distance) for v in pinhole_vals) / len(pinhole_vals) if pinhole_vals else None
            err_pitch   = sum(abs(v - real_distance) for v in pitch_vals) / len(pitch_vals) if pitch_vals else None
            # err_new     = sum(abs(v - real_distance) for v in new_vals) / len(new_vals) if new_vals else None

            errors = {
                "pinhole": err_pinhole,
                "pitch": err_pitch,
                # "new": err_new
            }

            errors = {k: v for k, v in errors.items() if v is not None}

            estimator_winner = min(errors, key=errors.get) if errors else None

            # --- fusion comparison: average vs bearing intersection vs weighted average ---
            avg_vals = [v for k, v in distances.items() if "fused_avg" in k and v is not None]
            bi_vals  = [v for k, v in distances.items() if "fused_bi" in k and v is not None]
            weighted_avg_vals = [v for k, v in distances.items() if "fused_weighted_avg" in k and v is not None]

            err_avg = sum(abs(v - real_distance) for v in avg_vals) / len(avg_vals) if avg_vals else None
            err_bi  = sum(abs(v - real_distance) for v in bi_vals) / len(bi_vals) if bi_vals else None
            err_weighted_avg = (
                sum(abs(v - real_distance) for v in weighted_avg_vals) / len(weighted_avg_vals)
                if weighted_avg_vals else None
            )

            # Pick best fusion among the ones that exist
            fusion_errors = {
                "average": err_avg,
                "bearing_intersection": err_bi,
                "weighted_average": err_weighted_avg,
            }
            fusion_errors = {k: v for k, v in fusion_errors.items() if v is not None}
            fusion_winner = min(fusion_errors, key=fusion_errors.get) if fusion_errors else None

            def fmt(x):
                return f"{x:.2f}" if x is not None else "NA"

            # Include errors for each drone in the estimator_winner output
            print(
                f"Frame {frame_idx} | "
                f"best estimator: {estimator_winner or 'NA'} "
                f"(pinhole_d1={fmt(distances.get('d1_pinhole'))}, pinhole_d2={fmt(distances.get('d2_pinhole'))}, "
                f"pitch_d1={fmt(distances.get('d1_pitch'))}, pitch_d2={fmt(distances.get('d2_pitch'))}) | "
                f"best fusion: {fusion_winner or 'NA'} "
                f"(avg={fmt(err_avg)} bi={fmt(err_bi)} weighted_avg={fmt(err_weighted_avg)})"
            )

        return {
            "drone1_lat": self.camera_drone1.lat,
            "drone1_lon": self.camera_drone1.lon,
            "drone2_lat": self.camera_drone2.lat,
            "drone2_lon": self.camera_drone2.lon,
            "series": frame_series or {},
            "distances": distances,
        }


    def find_geoposition_by_bbox(self, detections, bbox):
        """Find geoposition information for a detection by its bounding box."""
        if not detections or not bbox:
            return None

        for det in detections:
            if det.get("bbox") == list(bbox) or det.get("bbox") == bbox:
                return det.get("geo_position_pinhole")

        return None