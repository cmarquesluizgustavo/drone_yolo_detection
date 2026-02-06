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

class DualDroneDetectionPipeline:
    """Pipeline for processing synchronized video streams from two drones."""

    def __init__(self, model_path: str, person_confidence_threshold: float,
                 enable_weapon_detection: bool = True,
                 weapon_confidence_threshold: float = 0.5,
                 sample_majority_threshold: int = 1,
                 association_threshold: float = 100.0):  # distance threshold for cross-drone detection matching (meters)

        # Create two independent detection pipelines (one per drone)
        self.pipeline_drone1 = DetectionPipeline(
            model_path, 
            person_confidence_threshold=person_confidence_threshold,
            enable_weapon_detection=enable_weapon_detection,
            weapon_confidence_threshold=weapon_confidence_threshold,
            sample_majority_threshold=sample_majority_threshold
        )
        
        self.pipeline_drone2 = DetectionPipeline(
            model_path,
            person_confidence_threshold=person_confidence_threshold,
            enable_weapon_detection=enable_weapon_detection,
            weapon_confidence_threshold=weapon_confidence_threshold,
            sample_majority_threshold=sample_majority_threshold
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

    def print_console_output(self, detections1, detections2, fused_detections, weapon_results1=None, weapon_results2=None):
        """Print console output with detection information."""
        
        # Select best detections
        best1 = detections1[0] if detections1 else None
        best2 = detections2[0] if detections2 else None
        best_fused = fused_detections[0] if fused_detections else None

        d1_h = best1.get('distance_pinhole_m') if best1 else None
        d1_t = best1.get('distance_pitch_m') if best1 else None
        d2_h = best2.get('distance_pinhole_m') if best2 else None
        d2_t = best2.get('distance_pitch_m') if best2 else None

        geo1 = best1.get('person_geoposition') if best1 else None
        geo2 = best2.get('person_geoposition') if best2 else None

        geo_fused = best_fused.get('fused_geoposition') if best_fused else None
        d1_fused = distance_from_geoposition(self.camera_drone1, geo_fused['latitude'], geo_fused['longitude']) if geo_fused else None
        d2_fused = distance_from_geoposition(self.camera_drone2, geo_fused['latitude'], geo_fused['longitude']) if geo_fused else None

        c1 = best1.get('person_confidence') if best1 else None
        c2 = best2.get('person_confidence') if best2 else None
        c_fused = best_fused.get('person_confidence') if best_fused else None

        def best_weapon_conf(weapon_results):
            if not weapon_results:
                return None
            try:
                best = weapon_results[0] if len(weapon_results) > 0 else None
                if not (best and best.get('has_weapons') and best.get('weapon_detections')):
                    return 0.0
                return max([w.get('weapon_confidence', w.get('confidence', 0.0)) for w in best.get('weapon_detections', [])])
            except Exception:
                return None

        w1 = best_weapon_conf(weapon_results1)
        w2 = best_weapon_conf(weapon_results2)
        w_fused = best_fused.get('weapon_confidence') if best_fused else None

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
        
    def process_dual_drone_samples(self, input_dir_drone1, input_dir_drone2, output_base_dir):
        # Check if directories contain angle subdirectories
        angle_dirs_drone1 = [d for d in os.listdir(input_dir_drone1)
                            if os.path.isdir(os.path.join(input_dir_drone1, d)) and d.isdigit()]
        angle_dirs_drone2 = [d for d in os.listdir(input_dir_drone2)
                            if os.path.isdir(os.path.join(input_dir_drone2, d)) and d.isdigit()]
        
        # Find common angle directories
        common_angles = sorted(set(angle_dirs_drone1) & set(angle_dirs_drone2))
        
        if common_angles:
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
        
        # Process each synchronized frame pair
        for frame_idx, (frame1_path, frame2_path) in enumerate(synchronized_pairs):
            if frame_idx % 10 == 0:  # Print progress every 10 frames
                print(f"    Processing frame {frame_idx + 1}/{len(synchronized_pairs)}")
            self.process_frame_pair(
                frame1_path, frame2_path, frame_idx,
                sample_det_d1, sample_det_d2, sample_det_fused,
                sample_crops_d1, sample_crops_d2,
                sample_name
            )
        
        print(f"  ✓ Completed all {len(synchronized_pairs)} frames")
    
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
            return
        
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
        
        # Verbose output for fusion
        if self.verbose or (frame_idx % 10 == 0):
            fused_count = len([d for d in fused_detections if d.get('source') == 'fused'])
            print(f"      Frame {frame_idx}: Fused {fused_count} detections from both drones, total {len(fused_detections)} detections")

        # Print console output if needed
        if getattr(self, 'verbose', False):
            self.print_console_output(detections1, detections2, fused_detections, weapon_results1, weapon_results2)

        # Draw detections on individual frames
        img1_annotated = self.draw_boxes_fusion(img1, detections1, weapon_results1, "Drone 1")
        img2_annotated = self.draw_boxes_fusion(img2, detections2, weapon_results2, "Drone 2")
        
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
        if best_det1 and weapon_results1:
            best_idx1 = detections1.index(best_det1)
            if best_idx1 < len(weapon_results1) and weapon_results1[best_idx1].get('has_weapons', False):
                weapons_detected_d1_best = len(weapon_results1[best_idx1].get('weapon_detections', []))
                people_with_weapons_d1 = 1
        
        weapons_detected_d2_best = 0
        people_with_weapons_d2 = 0
        if best_det2 and weapon_results2:
            best_idx2 = detections2.index(best_det2)
            if best_idx2 < len(weapon_results2) and weapon_results2[best_idx2].get('has_weapons', False):
                weapons_detected_d2_best = len(weapon_results2[best_idx2].get('weapon_detections', []))
                people_with_weapons_d2 = 1
        
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
        )
        
          # No console logging
    
    def draw_boxes_fusion(self, image, detections, weapon_results, drone_label):
        tracks = viewer.tracks_from_detections(detections, weapon_results, track_id_start=1)
        img_annotated = viewer.draw_bbox(image, tracks, show_confidence=self.show_weapon_confidence)

        # Keep a simple top label for the side-by-side fused visualization.
        cv2.putText(img_annotated, drone_label, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        return img_annotated
    
    def fuse_frame_detections(self, detections1, detections2, weapon_results1, weapon_results2, frame_idx):
        """Fuse detections from two drones using direct geometric triangulation."""
        
        # Convert to Detection objects for fusion
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
        
        # Prepare measurements for direct triangulation
        measurement_groups = self.fusion.prepare_measurements_for_triangulation(
            detection_objs1, detection_objs2, self.camera_drone1, self.camera_drone2
        )
        
        # Use direct geometric triangulation (no temporal filtering)
        fused_detections = self._direct_triangulation(measurement_groups)
        
        # Convert triangulation results to detection format
        fused = []
        for detection in fused_detections:
            # Get triangulated position
            x, y = detection['x'], detection['y']
            
            # Convert back to geographic coordinates
            lat, lon = GeoConverter.xy_to_geo(x, y, ref_lat=self.camera_drone1.lat, ref_lon=self.camera_drone1.lon)
            
            # Determine source and bbox info
            source = 'fused' if len(detection['drone_measurements']) > 1 else f"drone{detection.get('drone_id', 1)}"
            
            fused_det = {
                'source': source,
                'person_confidence': detection['person_confidence'],
                'has_weapon': detection['has_weapon'],
                'weapon_confidence': detection['weapon_confidence'],
                'fused_geoposition': {
                    'latitude': lat,
                    'longitude': lon
                },
                'x': x,
                'y': y,
                'detection_id': detection['detection_id'],
            }
            
            # Add bounding box information
            if source == 'fused':
                fused_det['bbox_drone1'] = detection.get('bbox_drone1')
                fused_det['bbox_drone2'] = detection.get('bbox_drone2')
            else:
                bbox_key = f'bbox_drone{detection.get("drone_id", 1)}'
                fused_det['bbox'] = detection.get(bbox_key)
            
            # Calculate distance and bearing from each drone to fused position
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

        # Viewer-style info panels (translucent background + PIL text)
        panel_scale = max(0.7, min(1.6, max_h / 720.0))

        # Panel 1: fused detections summary
        fused_lines = [("FUSED DETECTIONS", viewer.color_text_title)]
        max_lines = 12
        for i, det in enumerate(fused_detections[:max_lines]):
            source = det.get('source', 'unknown')
            conf = det.get('person_confidence', det.get('confidence', 0.0))
            has_weapon = bool(det.get('has_weapon', False))
            weapon_conf = float(det.get('weapon_confidence', 0.0) or 0.0)
            distance = det.get('distance_m')

            info_text = f"#{i+1}: {source} | Conf={conf:.3f}"
            if distance is not None and distance > 0:
                info_text += f" | Dist={distance:.1f}m"
            if has_weapon:
                info_text += f" | ARMADO={weapon_conf:.3f}"
            if det.get('source') == 'fused' and det.get('fused_geoposition'):
                try:
                    lat = float(det['fused_geoposition']['latitude'])
                    lon = float(det['fused_geoposition']['longitude'])
                    info_text += f" | GEO={lat:.6f},{lon:.6f}"
                except Exception:
                    pass

            line_color = viewer.color_text_weapon if has_weapon else viewer.color_text_body
            fused_lines.append((info_text, line_color))

        if len(fused_detections) > max_lines:
            fused_lines.append((f"... ({len(fused_detections) - max_lines} more)", viewer.color_text_body))

        combined, panel1_rect = viewer.draw_info_panel(
            combined,
            fused_lines,
            x=10,
            y=50,
            scale_factor=panel_scale,
            align='left',
            bg_color=(0, 0, 0),
            bg_opacity=0.78,
        )

        # Panel 2: geo comparison (per-drone)
        geo_lines = [("GEO COMPARISON (per-drone)", viewer.color_text_title)]
        max_geo_lines = 6
        for i, det in enumerate(fused_detections[:max_geo_lines]):
            bbox1 = det.get('bbox_drone1') if det.get('source') == 'fused' else (det.get('bbox') if det.get('source') == 'drone1' else None)
            bbox2 = det.get('bbox_drone2') if det.get('source') == 'fused' else (det.get('bbox') if det.get('source') == 'drone2' else None)

            geo1 = self.find_geoposition_by_bbox(detections1, bbox1) if detections1 and bbox1 else None
            geo2 = self.find_geoposition_by_bbox(detections2, bbox2) if detections2 and bbox2 else None

            geo1_str = f"({geo1.get('latitude', 0):.6f},{geo1.get('longitude', 0):.6f})" if geo1 and isinstance(geo1, dict) else "None"
            geo2_str = f"({geo2.get('latitude', 0):.6f},{geo2.get('longitude', 0):.6f})" if geo2 and isinstance(geo2, dict) else "None"
            geo_text = f"#{i+1}: D1={geo1_str} | D2={geo2_str}"
            geo_lines.append((geo_text, viewer.color_text_body))

        p1_x1, p1_y1, p1_x2, p1_y2 = panel1_rect
        geo_y = min(max_h - 10, p1_y2 + int(12 * panel_scale))
        combined, _ = viewer.draw_info_panel(
            combined,
            geo_lines,
            x=10,
            y=int(geo_y),
            scale_factor=panel_scale,
            align='left',
            bg_color=(0, 0, 0),
            bg_opacity=0.72,
        )
        

        # Bottom summary panel
        total_dets = len(fused_detections)
        total_weapons = sum(1 for d in fused_detections if d.get('has_weapon', False))
        avg_conf = np.mean([d.get('person_confidence', d.get('confidence', 0.0)) for d in fused_detections]) if fused_detections else 0.0
        summary_text = f"Total: {total_dets} persons, {total_weapons} armed | Avg fused conf: {avg_conf:.3f}"
        combined, _ = viewer.draw_info_panel(
            combined,
            [(summary_text, viewer.color_text_body)],
            x=10,
            y=int(max_h - int(45 * panel_scale)),
            scale_factor=panel_scale,
            align='left',
            bg_color=(0, 0, 0),
            bg_opacity=0.70,
        )

        # Titles (viewer style)
        combined, _ = viewer.draw_info_panel(
            combined,
            [("Drone 1", viewer.color_text_title)],
            x=int(w1 // 2),
            y=10,
            scale_factor=panel_scale,
            align='center',
            bg_color=(0, 0, 0),
            bg_opacity=0.65,
        )
        combined, _ = viewer.draw_info_panel(
            combined,
            [("Drone 2", viewer.color_text_title)],
            x=int(w1 + (w2 // 2)),
            y=10,
            scale_factor=panel_scale,
            align='center',
            bg_color=(0, 0, 0),
            bg_opacity=0.65,
        )
        
        return combined

    def detect_people_with_estimation(self, image, drone_id):
        """Detect people and estimate their distance/bearing/geoposition for fusion."""
        
        detector = self.pipeline_drone1.detector if drone_id == 1 else self.pipeline_drone2.detector
        camera = self.camera_drone1 if drone_id == 1 else self.camera_drone2
    
        # Run detection
        results = detector.model(
            image,
            imgsz=640,
            iou=0.6,
            conf=self.person_confidence_threshold,
            classes=[0],
            verbose=False,
        )

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
