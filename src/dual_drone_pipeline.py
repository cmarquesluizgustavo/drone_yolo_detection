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
    # Position fusion method: 'average', 'rays', or 'circles'
    self.position_fusion_method = 'average'

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
        best_fused = fused_detections[0] if fused_detections else None

        d1_h = best_det1.get('distance_pinhole_m') if best_det1 else None
        d1_t = best_det1.get('distance_pitch_m') if best_det1 else None
        d2_h = best_det2.get('distance_pinhole_m') if best_det2 else None
        d2_t = best_det2.get('distance_pitch_m') if best_det2 else None

        geo1 = best_det1.get('person_geoposition') if best_det1 else None
        geo2 = best_det2.get('person_geoposition') if best_det2 else None

        geo_fused = best_fused.get('fused_geoposition') if best_fused else None
        d1_fused = distance_from_geoposition(self.camera_drone1, geo_fused['latitude'], geo_fused['longitude']) if geo_fused else None
        d2_fused = distance_from_geoposition(self.camera_drone2, geo_fused['latitude'], geo_fused['longitude']) if geo_fused else None

        c1 = best_det1.get('person_confidence') if best_det1 else None
        c2 = best_det2.get('person_confidence') if best_det2 else None
        c_fused = best_fused.get('person_confidence') if best_fused else None

        w1 = w1_conf
        w2 = w2_conf
        # Use max weapon confidence across ALL fused detections to match
        # stats_fused which counts weapons_fused = sum(... has_weapon ...)
        # across all fused detections, not just the first one.
        w_fused = 0.0
        if fused_detections:
            fused_confs = [d.get('weapon_confidence', 0.0) or 0.0 for d in fused_detections]
            if fused_confs:
                w_fused = max(fused_confs)
            if w_fused == 0.0:
                w_fused = None

        # Format inline
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
            f"geo2={fmt_geo(geo2)}, "
            f"d1_fused={fmt(d1_fused)}, "
            f"d2_fused={fmt(d2_fused)}, "
            f"geo_fused={fmt_geo(geo_fused)}"
        )
        print(
            "DETECTIONS: "
            f"c1={fmt(c1)}, "
            f"c2={fmt(c2)}, "
            f"c_fused={fmt(c_fused)}, "
            f"w1={fmt(w1)}, "
            f"w2={fmt(w2)}, "
            f"w_fused={fmt(w_fused)}"
        )
    
    def add_fused_geopositions(self, fused_detections, detections1, detections2):
        """Legacy method - geopositions are now added by direct triangulation in fuse_frame_detections."""
        # This method is kept for backward compatibility but is no longer used
        # Direct triangulation handles all position estimation
        pass
        
    def process_dual_drone_samples(self, input_dir_drone1, input_dir_drone2, output_base_dir, only_angle=None):
        # Check if directories contain angle subdirectories
        angle_dirs_drone1 = [d for d in os.listdir(input_dir_drone1)
                            if os.path.isdir(os.path.join(input_dir_drone1, d)) and d.isdigit()]
        angle_dirs_drone2 = [d for d in os.listdir(input_dir_drone2)
                            if os.path.isdir(os.path.join(input_dir_drone2, d)) and d.isdigit()]
        
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
            print(f"Processing each angle separately...")
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
                        set(getattr(self.stats_drone1, 'distance_pairs_by_dist_height', {}).keys())
                        | set(getattr(self.stats_drone2, 'distance_pairs_by_dist_height', {}).keys())
                        | set(getattr(self.stats_fused, 'distance_pairs_fused_d1_by_dist_height', {}).keys())
                        | set(getattr(self.stats_fused, 'distance_pairs_fused_d2_by_dist_height', {}).keys())
                    )

                    if combos:
                        print("\nFUSION GAIN BY (DISTANCE, HEIGHT):")
                        for (dist, height) in combos:
                            print(f"   Distance: {dist}m, Height: {height}m")

                            base_pairs_d1 = getattr(self.stats_drone1, 'distance_pairs_by_dist_height', {}).get((dist, height))
                            fused_pairs_d1 = getattr(self.stats_fused, 'distance_pairs_fused_d1_by_dist_height', {}).get((dist, height))
                            base_pairs_d2 = getattr(self.stats_drone2, 'distance_pairs_by_dist_height', {}).get((dist, height))
                            fused_pairs_d2 = getattr(self.stats_fused, 'distance_pairs_fused_d2_by_dist_height', {}).get((dist, height))

                            rmse_base_d1 = self.stats_drone1.compute_rmse(base_pairs_d1) if base_pairs_d1 else None
                            rmse_fused_d1_b = self.stats_fused.compute_rmse(fused_pairs_d1) if fused_pairs_d1 else None
                            rmse_base_d2 = self.stats_drone2.compute_rmse(base_pairs_d2) if base_pairs_d2 else None
                            rmse_fused_d2_b = self.stats_fused.compute_rmse(fused_pairs_d2) if fused_pairs_d2 else None

                            # Same formatting as overall so logs can be parsed consistently.
                            _print_gain("D1", rmse_base_d1, rmse_fused_d1_b)
                            _print_gain("D2", rmse_base_d2, rmse_fused_d2_b)
                except Exception:
                    # Keep stats printing robust.
                    pass
        else:
            # No angle subdirectories, process directly
            self._process_angle_directory(input_dir_drone1, input_dir_drone2, output_base_dir, None)
            for s in (self.stats_drone1, self.stats_drone2, self.stats_fused):
                s.finalize()
    
    def _process_angle_directory(self, input_dir_drone1, input_dir_drone2, output_base_dir, angle=None):
        """Process a single angle directory or direct sample directory."""
        # Get sample directories from both drones
        samples_drone1 = [d for d in os.listdir(input_dir_drone1)
                             if os.path.isdir(os.path.join(input_dir_drone1, d))]
        samples_drone2 = [d for d in os.listdir(input_dir_drone2)
                             if os.path.isdir(os.path.join(input_dir_drone2, d))]
        
        # Find matching sample pairs (same sample name in both directories)
        sample_pairs = []
        for sample1 in samples_drone1:
            if sample1 in samples_drone2:
                sample_pairs.append(sample1)
        
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
            
            # Get paths
            sample_path_drone1 = os.path.join(input_dir_drone1, sample_name)
            sample_path_drone2 = os.path.join(input_dir_drone2, sample_name)
            
            # Determine ground truth from filename convention when possible.
            sample_meta = {}
            try:
                sample_meta = self.pipeline_drone1.detector.extract_filename_metadata(sample_name)
            except Exception:
                sample_meta = {}
            if (sample_meta or {}).get('sample_class') in ('real', 'falso'):
                sample_ground_truth = (sample_meta.get('sample_class') == 'real')
                sample_class = sample_meta.get('sample_class')
            else:
                sample_ground_truth = sample_name.lower().startswith("real")
                sample_class = 'real' if sample_name.lower().startswith('real') else 'falso'
            
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
                fused_base_dir
            )
        
        # Finalize statistics for this angle
        for s in (self.stats_drone1, self.stats_drone2, self.stats_fused):
            s.finalize()

    def process_synchronized_sample_pair(self, sample_dir_drone1, sample_dir_drone2, output_base, sample_name, detections_base, crops_base, fused_base):
        # Supported image formats
        SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
        
        def get_image_files(directory):
            return sorted([
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.splitext(f)[1].lower() in [ext.lower() for ext in SUPPORTED_FORMATS]
            ])
        
        images_drone1 = get_image_files(sample_dir_drone1)
        images_drone2 = get_image_files(sample_dir_drone2)
        
        print(f"  Drone 1: {len(images_drone1)} frames")
        print(f"  Drone 2: {len(images_drone2)} frames")

        synchronized_pairs = self.synchronize_by_frame_index(
            images_drone1, images_drone2
        )
        
        print(f"  Synchronized: {len(synchronized_pairs)} frame pairs")
        
        sample_det_d1 = os.path.join(detections_base, f"{sample_name}_drone1")
        sample_det_d2 = os.path.join(detections_base, f"{sample_name}_drone2")
        sample_det_fused = os.path.join(fused_base, sample_name)
        sample_crops_d1 = os.path.join(crops_base, f"{sample_name}_drone1")
        sample_crops_d2 = os.path.join(crops_base, f"{sample_name}_drone2")
        
        for dir_path in [sample_det_d1, sample_det_d2, sample_det_fused,
                        sample_crops_d1, sample_crops_d2]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        ref_lat: float | None = None
        ref_lon: float | None = None
        
        # Process each synchronized frame pair
        for frame_idx, (frame1_path, frame2_path) in enumerate(synchronized_pairs):
            if frame_idx % 10 == 0:  # Print progress every 10 frames
                print(f"    Processing frame {frame_idx + 1}/{len(synchronized_pairs)}")
            frame_result = self.process_frame_pair(
                frame1_path, frame2_path, frame_idx,
                sample_det_d1, sample_det_d2, sample_det_fused,
                sample_crops_d1, sample_crops_d2,
                sample_name
            )

            if self.enable_ground_plane_plot and isinstance(frame_result, dict):
                if ref_lat is None or ref_lon is None:
                    try:
                        ref_lat = float(frame_result.get('drone1_lat'))
                        ref_lon = float(frame_result.get('drone1_lon'))
                    except Exception:
                        ref_lat, ref_lon = None, None
                try:
                    series_bundle = frame_result.get('series', None)
                    if series_bundle and isinstance(series_bundle, dict):
                        # GPS plot
                        gps = series_bundle.get('gps')
                        if gps and ref_lat is not None and ref_lon is not None:
                            out_path = os.path.join(sample_det_fused, f"{sample_name}_frame_{frame_idx:04d}_ground_gps.png")
                            plot_ground_plane(
                                ref_lat=float(ref_lat),
                                ref_lon=float(ref_lon),
                                drone_positions=gps.get('drone_positions'),
                                targets_d1=gps.get('targets_d1'),
                                targets_d2=gps.get('targets_d2'),
                                targets_fused=gps.get('targets_fused'),
                                measurements_d1=gps.get('measurements_d1'),
                                measurements_d2=gps.get('measurements_d2'),
                                title=f"{sample_name} – frame {frame_idx:04d} (GPS)",
                                out_path=out_path,
                                show=False,
                            )
                except Exception as e:
                    # Keep pipeline robust if plotting fails, but don't fail silently.
                    if getattr(self, 'verbose', False):
                        print(f"    ⚠ Ground-plane plot failed on frame {frame_idx:04d}: {e}")
        
        print(f"  ✓ Completed all {len(synchronized_pairs)} frames")

        # Per-frame plots are written during processing when enabled.
    
    @staticmethod
    def synchronize_by_frame_index(frames1, frames2):
        """Synchronize frames from two drones by extracting frame indices."""
        def extract_frame_num(path):
            import re
            basename = path.split('/')[-1].split('\\')[-1]
            matches = re.findall(r'_(\d{4})', basename)
            if matches:
                return int(matches[-1])
            matches = re.findall(r'(\d{4})', basename)
            if matches:
                return int(matches[-1])
            return -1
        
        # Build frame index maps
        frames1_dict = {extract_frame_num(f): f for f in frames1}
        frames2_dict = {extract_frame_num(f): f for f in frames2}
        
        # Find common frame indices
        common_indices = sorted(set(frames1_dict.keys()) & set(frames2_dict.keys()))
        
        synchronized_pairs = [(frames1_dict[idx], frames2_dict[idx]) for idx in common_indices]
        
        return synchronized_pairs

    def process_frame_pair(self, frame1_path, frame2_path, frame_idx, output_det_d1, output_det_d2, output_fused, output_crops_d1, output_crops_d2, sample_name):
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
            print(f"  DEBUG Drone positions:")
            print(f"    Drone 1: lat={self.camera_drone1.lat:.6f}, lon={self.camera_drone1.lon:.6f}, yaw={self.camera_drone1.yaw_deg:.1f}°")
            print(f"    Drone 2: lat={self.camera_drone2.lat:.6f}, lon={self.camera_drone2.lon:.6f}, yaw={self.camera_drone2.yaw_deg:.1f}°")
        
        # Extract height from filename metadata
        frame_meta = self.pipeline_drone1.detector.extract_filename_metadata(frame1_path)
        if frame_meta.get('height_m'):
            self.camera_drone1.height_m = frame_meta['height_m']
            self.camera_drone2.height_m = frame_meta['height_m']
        
        # Get filenames
        filename1 = os.path.basename(frame1_path)
        filename2 = os.path.basename(frame2_path)
        name1, ext1 = os.path.splitext(filename1)
        name2, ext2 = os.path.splitext(filename2)
        
        # Detect people using current estimation methods
        detections1 = self.detect_people_with_estimation(img1, drone_id=1)
        detections2 = self.detect_people_with_estimation(img2, drone_id=2)

        # Optional: filter known background people before crop/weapon detection and fusion.
        detections1 = self.pipeline_drone1.filter_people_detections(
            detections1,
            img1.shape,
            sample_name=sample_name,
            frame_path=frame1_path,
        )
        detections2 = self.pipeline_drone2.filter_people_detections(
            detections2,
            img2.shape,
            sample_name=sample_name,
            frame_path=frame2_path,
        )
        
        # Verbose output for detections
        if self.verbose or (frame_idx % 10 == 0):
            print(f"      Frame {frame_idx}: D1={len(detections1)} people, D2={len(detections2)} people")
        
        # Testing-time condition metadata (used for RMSE comparisons only)
        real_distance = frame_meta.get('distance_m')
        cam_height_m = frame_meta.get('height_m')
        
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
        
        # Fuse detections across drones using fused confidence
        fused_detections = self.fuse_frame_detections(
            detections1, detections2, weapon_results1, weapon_results2, frame_idx
        )

        # Ground-plane points for this frame (optional).
        frame_series = None
        if self.enable_ground_plane_plot:
            try:
                frame_series = build_series_from_frame(
                    drone1_lat=float(self.camera_drone1.lat),
                    drone1_lon=float(self.camera_drone1.lon),
                    drone2_lat=float(self.camera_drone2.lat),
                    drone2_lon=float(self.camera_drone2.lon),
                    detections1=detections1,
                    detections2=detections2,
                    fused_detections=[d for d in (fused_detections or []) if isinstance(d, dict) and d.get('source') == 'fused'],
                )
            except Exception:
                frame_series = None
        
        # Verbose output for fusion
        if self.verbose or (frame_idx % 10 == 0):
            fused_count = len([d for d in fused_detections if d.get('source') == 'fused'])
            print(f"      Frame {frame_idx}: Fused {fused_count} detections from both drones, total {len(fused_detections)} detections")

        # Draw detections on individual frames (with fused stats in each bbox)
        img1_annotated = self.draw_boxes_fusion(
            img1, detections1, weapon_results1, "Drone 1",
            fused_detections=fused_detections, drone_id=1, other_detections=detections2,
        )
        img2_annotated = self.draw_boxes_fusion(
            img2, detections2, weapon_results2, "Drone 2",
            fused_detections=fused_detections, drone_id=2, other_detections=detections1,
        )
        
        # Save individual drone detections
        cv2.imwrite(os.path.join(output_det_d1, f"{name1}_detected{ext1}"), img1_annotated)
        cv2.imwrite(os.path.join(output_det_d2, f"{name2}_detected{ext2}"), img2_annotated)
        
        # Create visualization of fused detections (side-by-side view)
        fused_vis = self.fused_visualization(
            img1_annotated, img2_annotated, fused_detections, detections1, detections2
        )
        
        # Save fused visualization if available
        fused_filename = f"{sample_name}_frame_{frame_idx:04d}_fused.jpg"
        fused_path = os.path.join(output_fused, fused_filename)
        if fused_vis is not None:
            cv2.imwrite(fused_path, fused_vis)
        
        # Update statistics
        try:
            sample_meta = self.pipeline_drone1.detector.extract_filename_metadata(sample_name)
        except Exception:
            sample_meta = {}
        if (sample_meta or {}).get('sample_class') in ('real', 'falso'):
            has_weapons_gt = (sample_meta.get('sample_class') == 'real')
        else:
            has_weapons_gt = sample_name.lower().startswith("real")
        weapons_detected_d1 = sum(1 for r in weapon_results1 if r.get('has_weapons', False))
        weapons_detected_d2 = sum(1 for r in weapon_results2 if r.get('has_weapons', False))
        
        # Count fused weapon detections
        weapons_fused = sum(1 for d in fused_detections if d.get('has_weapon', False))
        
        # Update stats for both individual detections and fused
        sample_class = (sample_meta.get('sample_class') if (sample_meta or {}).get('sample_class') in ('real', 'falso') else ('real' if has_weapons_gt else 'falso'))

        # Select CLOSEST detection from each drone (smallest distance)
        best_det1 = None
        if detections1:
            det1_with_dist = [d for d in detections1 if d.get('distance_m') is not None]
            if det1_with_dist:
                sorted_d1 = sorted(det1_with_dist, key=lambda d: d.get('distance_m', float('inf')))
                best_det1 = sorted_d1[0]
            else:
                # Fallback: use highest confidence
                best_det1 = max(detections1, key=lambda d: d.get('person_confidence', d.get('confidence', 0)))
        
        best_det2 = None
        if detections2:
            det2_with_dist = [d for d in detections2 if d.get('distance_m') is not None]
            if det2_with_dist:
                sorted_d2 = sorted(det2_with_dist, key=lambda d: d.get('distance_m', float('inf')))
                best_det2 = sorted_d2[0]
            else:
                # Fallback: use highest confidence
                best_det2 = max(detections2, key=lambda d: d.get('person_confidence', d.get('confidence', 0)))
        
        # Distance evaluation pairs (method-specific) - ONLY BEST DETECTION
        distances1 = []
        distances2 = []
        if best_det1 and best_det1.get('distance_m') is not None:
            distances1.append(best_det1['distance_m'])
        if best_det2 and best_det2.get('distance_m') is not None:
            distances2.append(best_det2['distance_m'])

        pairs1_p = []
        pairs1_pitch = []
        pairs2_p = []
        pairs2_pitch = []
        pairs1_primary = []
        pairs2_primary = []

        if real_distance is not None:
            if best_det1:
                if best_det1.get('distance_m') is not None:
                    pairs1_primary.append((best_det1['distance_m'], real_distance))
                if best_det1.get('distance_pinhole_m') is not None:
                    pairs1_p.append((best_det1['distance_pinhole_m'], real_distance))
                if best_det1.get('distance_pitch_m') is not None:
                    pairs1_pitch.append((best_det1['distance_pitch_m'], real_distance))
            if best_det2:
                if best_det2.get('distance_m') is not None:
                    pairs2_primary.append((best_det2['distance_m'], real_distance))
                if best_det2.get('distance_pinhole_m') is not None:
                    pairs2_p.append((best_det2['distance_pinhole_m'], real_distance))
                if best_det2.get('distance_pitch_m') is not None:
                    pairs2_pitch.append((best_det2['distance_pitch_m'], real_distance))
        
        # Weapon detection for best detection only
        weapons_detected_d1_best = 0
        people_with_weapons_d1 = 0
        w1_conf = 0.0
        if best_det1 and weapon_results1:
            best_idx1 = detections1.index(best_det1)
            if best_idx1 < len(weapon_results1):
                wr1 = weapon_results1[best_idx1]
                if wr1.get('has_weapons', False):
                    weapons_detected_d1_best = len(wr1.get('weapon_detections', []))
                    people_with_weapons_d1 = 1
                # Max weapon confidence for this person (matches what analyze_log reads)
                wdets1 = wr1.get('weapon_detections', [])
                if wdets1:
                    w1_conf = max(w.get('weapon_confidence', w.get('confidence', 0.0)) for w in wdets1)
        
        weapons_detected_d2_best = 0
        people_with_weapons_d2 = 0
        w2_conf = 0.0
        if best_det2 and weapon_results2:
            best_idx2 = detections2.index(best_det2)
            if best_idx2 < len(weapon_results2):
                wr2 = weapon_results2[best_idx2]
                if wr2.get('has_weapons', False):
                    weapons_detected_d2_best = len(wr2.get('weapon_detections', []))
                    people_with_weapons_d2 = 1
                # Max weapon confidence for this person (matches what analyze_log reads)
                wdets2 = wr2.get('weapon_detections', [])
                if wdets2:
                    w2_conf = max(w.get('weapon_confidence', w.get('confidence', 0.0)) for w in wdets2)

        # Print console output using the same best-detection as stats
        if getattr(self, 'verbose', False):
            self.print_console_output(best_det1, best_det2, fused_detections, w1_conf, w2_conf)

        # Individual drone stats (using only best detection)
        self.stats_drone1.add_image_results(
            1 if best_det1 else 0, weapons_detected_d1_best, 
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
            1 if best_det2 else 0, weapons_detected_d2_best,
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

        # Fused-distance RMSE pairs: derive per-drone distance to fused geoposition.
        fused_pairs = []
        fused_pairs_d1 = []
        fused_pairs_d2 = []
        if real_distance is not None:
            for d in fused_detections:
                if d.get('source') != 'fused':
                    continue
                geo = d.get('fused_geoposition')
                if not (geo and isinstance(geo, dict)):
                    continue
                lat = geo.get('latitude')
                lon = geo.get('longitude')
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
            len(fused_detections), weapons_fused,
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

        # Return lightweight per-frame data for optional callers.
        if self.enable_ground_plane_plot:
            return {
                'drone1_lat': self.camera_drone1.lat,
                'drone1_lon': self.camera_drone1.lon,
                'drone2_lat': self.camera_drone2.lat,
                'drone2_lon': self.camera_drone2.lon,
                'series': frame_series or {},
            }

        return None
    
    def draw_boxes_fusion(self, image, detections, weapon_results, drone_label, fused_detections=None, drone_id=None, other_detections=None):
        """Draw detection bboxes with both drone-specific and fused stats inside each bbox.
        
        Args:
            image: frame image
            detections: per-drone detections list
            weapon_results: per-drone weapon results
            drone_label: e.g. "Drone 1"
            fused_detections: list of fused detection dicts (from fuse_frame_detections)
            drone_id: 1 or 2, which drone this frame belongs to
            other_detections: detections from the OTHER drone (for cross-reference info)
        """
        tracks = viewer.tracks_from_detections(detections, weapon_results, track_id_start=1)
        
        # Build extra_lines per track with fused stats
        tracks_extra_lines = {}
        if fused_detections and drone_id is not None:
            bbox_key = f'bbox_drone{drone_id}'
            for track_idx, det in enumerate(detections):
                det_bbox = det.get('bbox')
                if det_bbox is None:
                    continue
                
                # Find matching fused detection
                matched_fused = None
                for fd in fused_detections:
                    fused_bbox = fd.get(bbox_key) or fd.get('bbox')
                    if fused_bbox is not None and list(fused_bbox) == list(det_bbox):
                        matched_fused = fd
                        break
                
                if matched_fused is None:
                    continue
                
                extra = []
                # Separator
                extra.append(("--- Fused ---", viewer.color_text_title))
                
                # Fused person confidence
                fused_conf = matched_fused.get('person_confidence', 0.0)
                extra.append((f"Fused Conf: {fused_conf:.3f}", viewer.color_text_body))
                
                # Fused geoposition
                geo = matched_fused.get('fused_geoposition')
                if geo and isinstance(geo, dict):
                    flat = geo.get('latitude', 0.0)
                    flon = geo.get('longitude', 0.0)
                    extra.append((f"Fused Lat:{flat:.6f} Lon:{flon:.6f}", viewer.color_text_body))
                
                # Fused distance (from this drone to the fused position)
                fused_dist_key = f'distance_drone{drone_id}_m'
                fused_dist = matched_fused.get(fused_dist_key)
                if fused_dist is not None:
                    extra.append((f"Fused Dist: {fused_dist:.1f}m", viewer.color_text_body))
                
                # Fused weapon info
                fused_has_weapon = matched_fused.get('has_weapon', False)
                fused_weapon_conf = matched_fused.get('weapon_confidence', 0.0)
                if fused_has_weapon:
                    extra.append((f"Fused ARMADO Conf: {fused_weapon_conf:.3f}", viewer.color_text_weapon))
                
                tracks_extra_lines[track_idx] = extra
        
        img_annotated = viewer.draw_bbox(
            image, tracks,
            show_confidence=self.show_weapon_confidence,
            tracks_extra_lines=tracks_extra_lines,
        )

        # Keep a simple top label for the side-by-side fused visualization.
        cv2.putText(img_annotated, drone_label, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        return img_annotated
    
    def fuse_frame_detections(self, detections1, detections2, weapon_results1, weapon_results2, frame_idx):
        """Fuse detections from two drones using direct geometric triangulation."""
        
    # Convert to Detection objects for fusion (keep existing fields for matching)
        detection_objs1 = []
        for i, det in enumerate(detections1):
            # Get weapon information
            has_weapon = False
            weapon_conf = 0.0
            if i < len(weapon_results1):
                has_weapon = weapon_results1[i].get('has_weapons', False)
                if has_weapon and weapon_results1[i].get('weapon_detections'):
                    weapon_conf = max([w.get('weapon_confidence', w.get('confidence', 0.0)) for w in weapon_results1[i]['weapon_detections']])
            
            # Get geoposition from detection
            person_geo = det.get('person_geoposition') or {}
            person_lat = person_geo.get('latitude') if isinstance(person_geo, dict) else None
            person_lon = person_geo.get('longitude') if isinstance(person_geo, dict) else None

            # Convert to ground-plane coordinates for matching.
            if person_lat is None or person_lon is None:
                x, y = float('nan'), float('nan')
                person_lat = 0.0
                person_lon = 0.0
            else:
                x, y = GeoConverter.geo_to_xy(float(person_lat), float(person_lon))
            
            detection_objs1.append(Detection(
                bbox=tuple(det['bbox']),
                person_confidence=det['person_confidence'],
                distance_m=(det.get('distance_m') if det.get('distance_m') is not None else 0.0),
                bearing_deg=(det.get('bearing_deg') if det.get('bearing_deg') is not None else 0.0),
                x=x,
                y=y,
                lat=person_lat,
                lon=person_lon,
                drone_id=1,
                frame_id=frame_idx,
                has_weapon=has_weapon,
                weapon_confidence=weapon_conf
            ))
        
        detection_objs2 = []
        for i, det in enumerate(detections2):
            # Get weapon information
            has_weapon = False
            weapon_conf = 0.0
            if i < len(weapon_results2):
                has_weapon = weapon_results2[i].get('has_weapons', False)
                if has_weapon and weapon_results2[i].get('weapon_detections'):
                    weapon_conf = max([w.get('weapon_confidence', w.get('confidence', 0.0)) for w in weapon_results2[i]['weapon_detections']])
            
            # Get geoposition from detection
            person_geo = det.get('person_geoposition') or {}
            person_lat = person_geo.get('latitude') if isinstance(person_geo, dict) else None
            person_lon = person_geo.get('longitude') if isinstance(person_geo, dict) else None

            # Convert to ground-plane coordinates for matching.
            if person_lat is None or person_lon is None:
                x, y = float('nan'), float('nan')
                person_lat = 0.0
                person_lon = 0.0
            else:
                x, y = GeoConverter.geo_to_xy(float(person_lat), float(person_lon))
            
            detection_objs2.append(Detection(
                bbox=tuple(det['bbox']),
                person_confidence=det['person_confidence'],
                distance_m=(det.get('distance_m') if det.get('distance_m') is not None else 0.0),
                bearing_deg=(det.get('bearing_deg') if det.get('bearing_deg') is not None else 0.0),
                x=x,
                y=y,
                lat=person_lat,
                lon=person_lon,
                drone_id=2,
                frame_id=frame_idx,
                has_weapon=has_weapon,
                weapon_confidence=weapon_conf
            ))
        # Prepare measurement groups (used to pair detections across drones)
        measurement_groups = self.fusion.prepare_measurements_for_triangulation(
            detection_objs1, detection_objs2, self.camera_drone1, self.camera_drone2
        )

        import math

        fused = []
        for det_id, group in enumerate(measurement_groups):
            measurements = group.get('drone_measurements', [])
            if not measurements:
                continue

            # Compute per-measurement XY estimates (uav_x + r*sin(b), uav_y + r*cos(b))
            est_pts = []
            for m in measurements:
                uav_x, uav_y = m['uav_pos']
                r = float(m.get('distance', 0.0) or 0.0)
                b_rad = math.radians(float(m.get('bearing', 0.0) or 0.0))
                x = uav_x + r * math.sin(b_rad)
                y = uav_y + r * math.cos(b_rad)
                est_pts.append((x, y))

            fused_x = None
            fused_y = None

            method = getattr(self, 'position_fusion_method', 'average')
            if method == 'average' or len(measurements) == 1:
                # Simple average of estimated XY positions (or single-measurement estimate)
                xs = [p[0] for p in est_pts]
                ys = [p[1] for p in est_pts]
                fused_x = sum(xs) / len(xs)
                fused_y = sum(ys) / len(ys)

            elif method == 'rays' and len(measurements) >= 2:
                # Intersect bearing rays (use first two measurements)
                m1 = measurements[0]
                m2 = measurements[1]
                x1, y1 = m1['uav_pos']
                x2, y2 = m2['uav_pos']
                b1 = math.radians(float(m1.get('bearing', 0.0) or 0.0))
                b2 = math.radians(float(m2.get('bearing', 0.0) or 0.0))
                dx1, dy1 = math.sin(b1), math.cos(b1)
                dx2, dy2 = math.sin(b2), math.cos(b2)
                cross = dx1 * dy2 - dy1 * dx2
                if abs(cross) < 1e-6:
                    # Nearly parallel -> fallback to average
                    xs = [p[0] for p in est_pts]
                    ys = [p[1] for p in est_pts]
                    fused_x = sum(xs) / len(xs)
                    fused_y = sum(ys) / len(ys)
                else:
                    dx = x2 - x1
                    dy = y2 - y1
                    t1 = (dx * dy2 - dy * dx2) / cross
                    fused_x = x1 + t1 * dx1
                    fused_y = y1 + t1 * dy1

            elif method == 'circles' and len(measurements) >= 2:
                # Intersect range circles (use first two measurements)
                m1 = measurements[0]
                m2 = measurements[1]
                x1, y1 = m1['uav_pos']
                x2, y2 = m2['uav_pos']
                r1 = float(m1.get('distance', 0.0) or 0.0)
                r2 = float(m2.get('distance', 0.0) or 0.0)
                d = math.hypot(x2 - x1, y2 - y1)
                if d < 1e-6 or d > (r1 + r2) or d < abs(r1 - r2):
                    # No intersection or degenerate -> fallback to average
                    xs = [p[0] for p in est_pts]
                    ys = [p[1] for p in est_pts]
                    fused_x = sum(xs) / len(xs)
                    fused_y = sum(ys) / len(ys)
                else:
                    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
                    h = math.sqrt(max(0.0, r1 * r1 - a * a))
                    xm = x1 + a * (x2 - x1) / d
                    ym = y1 + a * (y2 - y1) / d
                    rx = -(y2 - y1) * (h / d)
                    ry = (x2 - x1) * (h / d)
                    xi1 = xm + rx
                    yi1 = ym + ry
                    xi2 = xm - rx
                    yi2 = ym - ry
                    # Choose the intersection closer to the average of estimates
                    avg_x = sum([p[0] for p in est_pts]) / len(est_pts)
                    avg_y = sum([p[1] for p in est_pts]) / len(est_pts)
                    d1 = (xi1 - avg_x) ** 2 + (yi1 - avg_y) ** 2
                    d2 = (xi2 - avg_x) ** 2 + (yi2 - avg_y) ** 2
                    if d1 <= d2:
                        fused_x, fused_y = xi1, yi1
                    else:
                        fused_x, fused_y = xi2, yi2

            else:
                # Unknown method - fallback to average
                xs = [p[0] for p in est_pts]
                ys = [p[1] for p in est_pts]
                fused_x = sum(xs) / len(xs)
                fused_y = sum(ys) / len(ys)

            # Convert XY to lat/lon
            lat, lon = GeoConverter.xy_to_geo(fused_x, fused_y)

            # Build fused detection
            fused_det = {
                'source': 'fused' if len(measurements) > 1 else f"drone{group.get('drone_id', 1)}",
                'person_confidence': group.get('person_confidence', 0.0),
                'has_weapon': group.get('has_weapon', False),
                'weapon_confidence': group.get('weapon_confidence', 0.0),
                'fused_geoposition': {'latitude': lat, 'longitude': lon},
                'x': fused_x,
                'y': fused_y,
                'detection_id': det_id + 1,
                'bbox_drone1': group.get('bbox_drone1'),
                'bbox_drone2': group.get('bbox_drone2'),
            }

            # Add per-drone distances (optional)
            for cam_id, camera in [(1, self.camera_drone1), (2, self.camera_drone2)]:
                try:
                    dist = distance_from_geoposition(camera, lat, lon)
                    fused_det[f'distance_drone{cam_id}_m'] = dist
                except Exception:
                    pass

            fused.append(fused_det)

        return fused
    
    def _direct_triangulation(self, measurement_groups):
        """
        Perform direct geometric triangulation for each frame independently.
        No temporal filtering - treats each frame as independent.
        """
        import math
        
        fused_detections = []
        for det_id, group in enumerate(measurement_groups):
            measurements = group['drone_measurements']
            
            if len(measurements) == 1:
                # Single drone measurement - use bearing line estimate
                m = measurements[0]
                bearing_rad = math.radians(m['bearing'])
                x = m['uav_pos'][0] + m['distance'] * math.sin(bearing_rad)
                y = m['uav_pos'][1] + m['distance'] * math.cos(bearing_rad)
                drone_id = group.get('drone_id', 1)
            elif len(measurements) >= 2:
                # Dual drone measurement - triangulate bearing line intersection
                m1, m2 = measurements[0], measurements[1]
                x1, y1 = m1['uav_pos']
                x2, y2 = m2['uav_pos']
                b1 = math.radians(m1['bearing'])
                b2 = math.radians(m2['bearing'])
                
                # Direction vectors
                dx1 = math.sin(b1)
                dy1 = math.cos(b1)
                dx2 = math.sin(b2)
                dy2 = math.cos(b2)
                
                # Check if parallel
                cross = dx1 * dy2 - dy1 * dx2
                if abs(cross) < 1e-6:
                    # Parallel - use weighted midpoint
                    r1, r2 = m1['distance'], m2['distance']
                    w1 = 1.0 / max(r1, 0.1)
                    w2 = 1.0 / max(r2, 0.1)
                    x1_est = x1 + r1 * dx1
                    y1_est = y1 + r1 * dy1
                    x2_est = x2 + r2 * dx2
                    y2_est = y2 + r2 * dy2
                    x = (x1_est * w1 + x2_est * w2) / (w1 + w2)
                    y = (y1_est * w1 + y2_est * w2) / (w1 + w2)
                else:
                    # Intersect bearing lines
                    dx = x2 - x1
                    dy = y2 - y1
                    t1 = (dx * dy2 - dy * dx2) / cross
                    x = x1 + t1 * dx1
                    y = y1 + t1 * dy1
                
                drone_id = None  # Fused from multiple drones
            
            fused_detections.append({
                'x': x,
                'y': y,
                'detection_id': det_id + 1,
                'person_confidence': group['person_confidence'],
                'has_weapon': group['has_weapon'],
                'weapon_confidence': group['weapon_confidence'],
                'bbox_drone1': group.get('bbox_drone1'),
                'bbox_drone2': group.get('bbox_drone2'),
                'drone_id': drone_id,
                'drone_measurements': measurements
            })
        
        return fused_detections

    def fused_visualization(self, img1, img2, fused_detections, detections1=None, detections2=None):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Make heights equal
        max_h = max(h1, h2)
        if h1 < max_h:
            img1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if h2 < max_h:
            img2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Concatenate horizontally
        combined = np.hstack([img1, img2])

        return combined

    def detect_people_with_estimation(self, image, drone_id):
        """Detect people and estimate their distance/bearing/geoposition for fusion."""
        
        detector = self.pipeline_drone1.detector if drone_id == 1 else self.pipeline_drone2.detector
        camera = self.camera_drone1 if drone_id == 1 else self.camera_drone2
    
        # Run detection
        infer_kwargs = dict(
            imgsz=640,
            iou=0.6,
            conf=self.person_confidence_threshold,
            classes=[0],
            verbose=False,
        )
        if getattr(detector, 'device', None) is not None:
            infer_kwargs['device'] = getattr(detector, 'device')

        results = detector.model(image, **infer_kwargs)

        detections_info = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                       
                    if class_id == 0 and confidence >= self.person_confidence_threshold:
                        person_height_px = float(y2 - y1)
                        y_bottom = float(y2)
                        x_center = float((x1 + x2) / 2)

                        # Always compute pinhole estimate
                        distance_pinhole_m = estimate_distance(camera, person_height_px) if person_height_px > 0 else None

                        distance_pitch_m = None
                        if camera.height_m > 0:
                            try:
                                distance_pitch_m = estimate_distance_pitch(camera, y_bottom)
                            except (ZeroDivisionError, ValueError, TypeError):
                                distance_pitch_m = None

                        # Calculate bearing from camera to target
                        bearing_deg = estimate_bearing(camera, x_center)

                        # Choose a primary distance for downstream fusion/triangulation.
                        distance_m = None
                        if distance_pitch_m is not None:
                            distance_m = float(distance_pitch_m)
                        elif distance_pinhole_m is not None:
                            distance_m = float(distance_pinhole_m)
                        
                        # Calculate geoposition of detected person (when telemetry lat/lon is available).
                        person_lat_pinhole, person_lon_pinhole = None, None
                        if distance_pinhole_m is not None and bearing_deg is not None and camera.lat != 0 and camera.lon != 0:
                            person_lat_pinhole, person_lon_pinhole = target_geoposition(camera, float(distance_pinhole_m), float(bearing_deg))

                        person_lat_pitch, person_lon_pitch = None, None
                        if distance_pitch_m is not None and bearing_deg is not None and camera.lat != 0 and camera.lon != 0:
                            person_lat_pitch, person_lon_pitch = target_geoposition(camera, float(distance_pitch_m), float(bearing_deg))

                        # A single geoposition used for matching (prefer pitch-based when available).
                        person_geoposition = None
                        if person_lat_pitch is not None and person_lon_pitch is not None:
                            person_geoposition = {'latitude': person_lat_pitch, 'longitude': person_lon_pitch}
                        elif person_lat_pinhole is not None and person_lon_pinhole is not None:
                            person_geoposition = {'latitude': person_lat_pinhole, 'longitude': person_lon_pinhole}



                        detections_info.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'person_confidence': confidence,
                            'distance_m': distance_m,
                            'distance_pinhole_m': distance_pinhole_m,
                            'distance_pitch_m': distance_pitch_m,
                            'bearing_deg': bearing_deg,
                            'cam_height_m': camera.height_m,
                            'cam_pitch_deg': camera.pitch_deg,  # Use actual telemetry
                            'cam_yaw_deg': camera.yaw_deg,
                            'cam_roll_deg': camera.roll_deg,
                            'cam_lat': camera.lat,
                            'cam_lon': camera.lon,
                            'person_geoposition': person_geoposition,
                            'geo_position_pinhole': {
                                'latitude': person_lat_pinhole,
                                'longitude': person_lon_pinhole
                            } if person_lat_pinhole is not None else None,
                            'geo_position_pitch': {
                                'latitude': person_lat_pitch,
                                'longitude': person_lon_pitch
                            } if person_lat_pitch is not None else None,
                        })
        return detections_info

    def find_geoposition_by_bbox(self, detections, bbox):
        """Find geoposition information for a detection by its bounding box."""
        if not detections or not bbox:
            return None
        
        for det in detections:
            if det.get('bbox') == list(bbox) or det.get('bbox') == bbox:
                return det.get('geo_position_pinhole')
        
        return None
