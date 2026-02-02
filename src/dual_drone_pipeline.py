import cv2
import os
import numpy as np
from pathlib import Path
from detection_pipeline import DetectionPipeline
from detection_fusion import DualDroneFusion, Detection
from camera import Camera

class DualDroneDetectionPipeline:
    """Pipeline for processing synchronized video streams from two drones."""

    def __init__(self, model_path: str, person_confidence_threshold: float,
                 enable_weapon_detection: bool = True,
                 weapon_confidence_threshold: float = 0.7,
                 sample_majority_threshold: int = 1,
                 association_threshold: float = 2.0):  # distance threshold for cross-drone detection matching (meters)

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
        
        # Share statistics between pipelines (use drone1's stats as primary)
        self.stats = self.pipeline_drone1.stats
        self.pipeline_drone2.stats = self.stats
        
        self.save_crops = True
        self.enable_weapon_detection = enable_weapon_detection
        self.person_confidence_threshold = person_confidence_threshold

    def print_console_output(self, detections1, detections2, fused_detections):
        """Print console output with detection information."""
        from position_estimation import distance_from_position
        
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
        d1_fused = distance_from_position(self.camera_drone1, geo_fused['latitude'], geo_fused['longitude']) if geo_fused else None
        d2_fused = distance_from_position(self.camera_drone2, geo_fused['latitude'], geo_fused['longitude']) if geo_fused else None

        c1 = best1.get('person_confidence') if best1 else None
        c2 = best2.get('person_confidence') if best2 else None
        c_fused = best_fused.get('person_confidence') if best_fused else None

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
            f"c_fused={fmt(c_fused)}"
        )
    
    def add_fused_geopositions(self, fused_detections, detections1, detections2):
        """Add triangulated geopositions to fused detections."""
        from position_estimation import fuse_target_geoposition
        
        for fused_det in fused_detections:
            if fused_det.get('source') == 'fused':
                # Find the original detections from both drones
                bbox1 = fused_det.get('bbox_drone1')
                bbox2 = fused_det.get('bbox_drone2')
                
                if bbox1 is None or bbox2 is None:
                    continue
                
                det1 = next((d for d in detections1 if tuple(d['bbox']) == tuple(bbox1)), None)
                det2 = next((d for d in detections2 if tuple(d['bbox']) == tuple(bbox2)), None)
                
                if det1 and det2:
                    dist1 = det1.get('distance_m')
                    bearing1 = det1.get('bearing_deg')
                    dist2 = det2.get('distance_m')
                    bearing2 = det2.get('bearing_deg')
                    
                    if all(v is not None for v in [dist1, bearing1, dist2, bearing2]):
                        # Triangulate fused position
                        fused_lat, fused_lon = fuse_target_geoposition(
                            self.camera_drone1, dist1, bearing1,
                            self.camera_drone2, dist2, bearing2
                        )
                        fused_det['fused_geoposition'] = {
                            'latitude': fused_lat,
                            'longitude': fused_lon
                        }
        
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
                self.stats.reset()
                self.stats._in_batch_mode = True
                
                # Process this angle
                self._process_angle_directory(angle_dir_drone1, angle_dir_drone2, angle_output_dir, angle)
                
                # Print statistics for this angle
                print(f"\n{'='*60}")
                print(f"STATISTICS FOR ANGLE {angle}°")
                print(f"{'='*60}")
                self.stats.print_summary()
        else:
            # No angle subdirectories, process directly
            self._process_angle_directory(input_dir_drone1, input_dir_drone2, output_base_dir, None)
            self.stats.finalize()
    
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
            print(f"Processing sample {sample_idx}/{len(sample_pairs)}: {sample_name}")
            
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
            
            # Mark start of new sample
            self.stats.start_new_sample(sample_ground_truth, sample_class)
            
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
        self.stats.finalize()

    def process_synchronized_sample_pair(self, sample_dir_drone1, sample_dir_drone2, output_base, sample_name, detections_base, crops_base, fused_base):
        from config import SUPPORTED_FORMATS
        
        def get_image_files(directory):
            return sorted([
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.splitext(f)[1].lower() in [ext.lower() for ext in SUPPORTED_FORMATS]
            ])
        
        images_drone1 = get_image_files(sample_dir_drone1)
        images_drone2 = get_image_files(sample_dir_drone2)

        synchronized_pairs = self.synchronize_by_frame_index(
            images_drone1, images_drone2
        )
        
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
            self.process_frame_pair(
                frame1_path, frame2_path, frame_idx,
                sample_det_d1, sample_det_d2, sample_det_fused,
                sample_crops_d1, sample_crops_d2,
                sample_name
            )
    
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
        
        # Load telemetry data for both cameras from txt files
        self.camera_drone1.load_telemetry_from_video_path(frame1_path)
        self.camera_drone2.load_telemetry_from_video_path(frame2_path)
        
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
        detections1 = self.detect_people_with_estimation(frame1_path, img1, drone_id=1)
        detections2 = self.detect_people_with_estimation(frame2_path, img2, drone_id=2)
        
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

        # Print console output if needed
        self.print_console_output(detections1, detections2, fused_detections)

        # Attach triangulated fused geoposition if implemented
        self.add_fused_geopositions(fused_detections, detections1, detections2)

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

        # Distance evaluation pairs (method-specific)
        distances1 = [d['distance_m'] for d in detections1 if d.get('distance_m') is not None]
        distances2 = [d['distance_m'] for d in detections2 if d.get('distance_m') is not None]

        pairs1 = []
        pairs2 = []
        pairs1_p = []
        pairs1_pitch = []
        pairs2_p = []
        pairs2_pitch = []

        if real_distance is not None:
            for d in detections1:
                if d.get('distance_m') is not None:
                    pairs1.append((d['distance_m'], real_distance))
                if d.get('distance_pinhole_m') is not None:
                    pairs1_p.append((d['distance_pinhole_m'], real_distance))
                if d.get('distance_pitch_m') is not None:
                    pairs1_pitch.append((d['distance_pitch_m'], real_distance))
            for d in detections2:
                if d.get('distance_m') is not None:
                    pairs2.append((d['distance_m'], real_distance))
                if d.get('distance_pinhole_m') is not None:
                    pairs2_p.append((d['distance_pinhole_m'], real_distance))
                if d.get('distance_pitch_m') is not None:
                    pairs2_pitch.append((d['distance_pitch_m'], real_distance))
        
        # Individual drone stats
        self.stats.add_image_results(
            len(detections1), weapons_detected_d1, 
            len([r for r in weapon_results1 if r.get('has_weapons', False)]),
            has_weapons_gt,
            distances=distances1,
            real_distance=real_distance,
            cam_height_m=cam_height_m,
            sample_class=sample_class,
            distance_pairs_pinhole=pairs1_p,
            distance_pairs_pitch=pairs1_pitch,
        )
        
        self.stats.add_image_results(
            len(detections2), weapons_detected_d2,
            len([r for r in weapon_results2 if r.get('has_weapons', False)]),
            has_weapons_gt,
            distances=distances2,
            real_distance=real_distance,
            cam_height_m=cam_height_m,
            sample_class=sample_class,
            distance_pairs_pinhole=pairs2_p,
            distance_pairs_pitch=pairs2_pitch,
        )
        
        # Fused detection stats (using fused confidence)
        self.stats.add_image_results(
            len(fused_detections), weapons_fused,
            weapons_fused,
            has_weapons_gt,
            distances=None,
            real_distance=None,
            cam_height_m=None,
            sample_class=sample_class,
            distance_pairs_pinhole=None,
            distance_pairs_pitch=None
        )
        
          # No console logging
    
    def draw_boxes_fusion(self, image, detections, weapon_results, drone_label):
        img_annotated = image.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            distance_m = det.get('distance_m')
            dist_p = det.get('distance_pinhole_m')
            dist_t = det.get('distance_pitch_m')
            
            # Check weapon detection
            has_weapon = False
            weapon_conf = 0.0
            if i < len(weapon_results):
                has_weapon = weapon_results[i].get('has_weapons', False)
                if has_weapon and weapon_results[i].get('weapon_detections'):
                    weapon_conf = max([w['confidence'] for w in weapon_results[i]['weapon_detections']])
            
            # Choose color based on weapon detection
            color = (0, 0, 255) if has_weapon else (0, 255, 0)  # Red if weapon, Green otherwise
            
            # Draw bounding box
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"Person: {confidence:.2f}"
            # Show both distance estimates when available for easy comparison.
            if dist_p is not None or dist_t is not None:
                if dist_p is not None:
                    label += f" P:{dist_p:.1f}m"
                if dist_t is not None:
                    label += f" Pitch:{dist_t:.1f}m"
            elif distance_m is not None:
                label += f" ({distance_m:.1f}m)"
            if has_weapon:
                label += f" WEAPON:{weapon_conf:.2f}"
            
            # Draw label
            cv2.putText(img_annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add drone label
        cv2.putText(img_annotated, drone_label, (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        return img_annotated
    
    def fuse_frame_detections(self, detections1, detections2, weapon_results1, weapon_results2, frame_idx):
        """Fuse detections from two drones using geoposition-based matching."""
        from geoconverter import GeoConverter
        from position_estimation import fuse_target_geoposition
        
        # Convert to Detection objects for fusion
        detection_objs1 = []
        for i, det in enumerate(detections1):
            # Get weapon information
            has_weapon = False
            weapon_conf = 0.0
            if i < len(weapon_results1):
                has_weapon = weapon_results1[i].get('has_weapons', False)
                if has_weapon and weapon_results1[i].get('weapon_detections'):
                    weapon_conf = max([w['confidence'] for w in weapon_results1[i]['weapon_detections']])
            
            # Get geoposition from detection
            person_geo = det.get('person_geoposition', {})
            person_lat = person_geo.get('latitude', 0.0) if person_geo else 0.0
            person_lon = person_geo.get('longitude', 0.0) if person_geo else 0.0
            
            # Convert to ground-plane coordinates for matching
            x, y = 0.0, 0.0
            if person_lat != 0.0 and person_lon != 0.0:
                x, y = GeoConverter.geo_to_xy(person_lat, person_lon)
            
            detection_objs1.append(Detection(
                bbox=tuple(det['bbox']),
                person_confidence=det['person_confidence'],
                distance_m=det.get('distance_m', 0.0),
                bearing_deg=det.get('bearing_deg', 0.0),
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
                    weapon_conf = max([w['confidence'] for w in weapon_results2[i]['weapon_detections']])
            
            # Get geoposition from detection
            person_geo = det.get('person_geoposition', {})
            person_lat = person_geo.get('latitude', 0.0) if person_geo else 0.0
            person_lon = person_geo.get('longitude', 0.0) if person_geo else 0.0
            
            # Convert to ground-plane coordinates for matching
            x, y = 0.0, 0.0
            if person_lat != 0.0 and person_lon != 0.0:
                x, y = GeoConverter.geo_to_xy(person_lat, person_lon)
            
            detection_objs2.append(Detection(
                bbox=tuple(det['bbox']),
                person_confidence=det['person_confidence'],
                distance_m=det.get('distance_m', 0.0),
                bearing_deg=det.get('bearing_deg', 0.0),
                x=x,
                y=y,
                lat=person_lat,
                lon=person_lon,
                drone_id=2,
                frame_id=frame_idx,
                has_weapon=has_weapon,
                weapon_confidence=weapon_conf
            ))
        
        # Perform fusion using geoposition-based matching
        fused = self.fusion.match_detections(detection_objs1, detection_objs2)
        
        # Add fused geopositions using triangulation
        for fused_det in fused:
            if fused_det.get('source') == 'fused':
                # Find the original detections from both drones
                bbox1 = fused_det.get('bbox_drone1')
                bbox2 = fused_det.get('bbox_drone2')
                
                det1 = next((d for d in detections1 if tuple(d['bbox']) == bbox1), None)
                det2 = next((d for d in detections2 if tuple(d['bbox']) == bbox2), None)
                
                if det1 and det2:
                    dist1 = det1.get('distance_m')
                    bearing1 = det1.get('bearing_deg')
                    dist2 = det2.get('distance_m')
                    bearing2 = det2.get('bearing_deg')
                    
                    if all([dist1, bearing1, dist2, bearing2]):
                        # Triangulate fused position
                        fused_lat, fused_lon = fuse_target_geoposition(
                            self.camera_drone1, dist1, bearing1,
                            self.camera_drone2, dist2, bearing2
                        )
                        fused_det['fused_geoposition'] = {
                            'latitude': fused_lat,
                            'longitude': fused_lon
                        }
        
        return fused

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
        
        # Add fusion information text overlay with fused confidence
        info_y = 60
        header_text = "=== FUSED DETECTIONS ==="
        cv2.putText(combined, header_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        info_y += 30
        
        for i, det in enumerate(fused_detections):
            source = det.get('source', 'unknown')
            conf = det.get('person_confidence', det.get('confidence', 0.0))  # fused confidence
            has_weapon = det.get('has_weapon', False)
            weapon_conf = det.get('weapon_confidence', 0.0)
            distance = det.get('distance_m')
            bearing = det.get('bearing_deg')
            
            # Create detailed info text
            info_text = f"#{i+1}: {source} | Conf={conf:.3f}"
            if distance is not None and distance > 0:
                info_text += f" | Dist={distance:.1f}m"
            if has_weapon:
                info_text += f" | WEAPON={weapon_conf:.3f}"
            if det.get('source') == 'fused' and det.get('fused_geoposition'):
                try:
                    lat = float(det['fused_geoposition']['latitude'])
                    lon = float(det['fused_geoposition']['longitude'])
                    info_text += f" | FUSED_GEO={lat:.6f},{lon:.6f}"
                except Exception:
                    pass
            
            # Color based on weapon detection
            text_color = (0, 0, 255) if has_weapon else (0, 255, 255)
            
            cv2.putText(combined, info_text, (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            info_y += 25

        # Geo comparison overlay (per-drone), mainly for debugging/consistency checks.
        info_y += 10
        geo_header_text = "=== GEO COMPARISON (per-drone) ==="
        cv2.putText(combined, geo_header_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        info_y += 28

        max_geo_lines = 6
        for i, det in enumerate(fused_detections[:max_geo_lines]):
            bbox1 = det.get('bbox_drone1') if det.get('source') == 'fused' else (det.get('bbox') if det.get('source') == 'drone1' else None)
            bbox2 = det.get('bbox_drone2') if det.get('source') == 'fused' else (det.get('bbox') if det.get('source') == 'drone2' else None)

            geo1 = self.find_geoposition_by_bbox(detections1, bbox1) if detections1 and bbox1 else None
            geo2 = self.find_geoposition_by_bbox(detections2, bbox2) if detections2 and bbox2 else None

            # Format geo inline
            geo1_str = f"({geo1.get('latitude', 0):.6f},{geo1.get('longitude', 0):.6f})" if geo1 and isinstance(geo1, dict) else "None"
            geo2_str = f"({geo2.get('latitude', 0):.6f},{geo2.get('longitude', 0):.6f})" if geo2 and isinstance(geo2, dict) else "None"
            geo_text = f"#{i+1}: D1={geo1_str} | D2={geo2_str}"

            cv2.putText(combined, geo_text, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            info_y += 22
        
        # Add summary at the bottom
        summary_y = max_h - 40
        total_dets = len(fused_detections)
        total_weapons = sum(1 for d in fused_detections if d.get('has_weapon', False))
        avg_conf = np.mean([d.get('person_confidence', d.get('confidence', 0.0)) for d in fused_detections]) if fused_detections else 0.0
        
        summary_text = f"Total: {total_dets} persons, {total_weapons} weapons | Avg Fused Conf: {avg_conf:.3f}"
        cv2.putText(combined, summary_text, (10, summary_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add title
        title_d1 = "Drone 1"
        title_d2 = "Drone 2"
        cv2.putText(combined, title_d1, (w1//2 - 50, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(combined, title_d2, (w1 + w2//2 - 50, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        return combined

    def detect_people_with_estimation(self, image_path, image, drone_id):
        """Detect people and estimate their positions using the appropriate drone's detector."""
        from position_estimation import estimate_distance, estimate_distance_2, estimate_bearing, simple_target_geoposition
        
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

        # Debug: check what the model returns (raw box count before confidence filtering)
        total_boxes = 0
        try:
            for result in results:
                if result.boxes is not None:
                    total_boxes += len(result.boxes)
        except Exception:
            total_boxes = 0

        if drone_id == 1 and total_boxes == 0:
            print(
                f"    DEBUG D{drone_id}: No detections - image shape: {image.shape}, "
                f"conf threshold: {self.person_confidence_threshold}, frame: {os.path.basename(image_path)}"
            )

        # Extract metadata from filename (for testing/comparison purposes)
        # NOTE: PeopleDetector.extract_filename_metadata returns keys: height_m, distance_m, sample_class
        file_data = detector.extract_filename_metadata(image_path)
        cam_height_m = file_data.get('height_m')
        cam_pitch_deg = file_data.get('pitch_deg') or file_data.get('camera_pitch_deg')

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

                        # Compute pitch-based estimate using telemetry data from txt file
                        distance_pitch_m = None
                        if camera.height_m > 0 and camera.pitch_deg != 0:
                            distance_pitch_m = estimate_distance_2(camera, y_bottom)

                        # Choose primary distance for downstream use
                        # Prefer pitch-based if available (more accurate with real telemetry)
                        if distance_pitch_m is not None:
                            distance_m = distance_pitch_m
                            distance_method = 'pitch_telemetry'
                        elif distance_pinhole_m is not None:
                            distance_m = distance_pinhole_m
                            distance_method = 'pinhole'
                        else:
                            distance_m = None
                            distance_method = None

                        # Calculate bearing from camera to target
                        bearing_deg = estimate_bearing(camera, x_center)
                        
                        # Calculate geoposition of detected person
                        person_lat, person_lon = None, None
                        if distance_m is not None and bearing_deg is not None and camera.lat != 0 and camera.lon != 0:
                            person_lat, person_lon = simple_target_geoposition(camera, distance_m, bearing_deg)

                        detections_info.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'person_confidence': confidence,
                            'distance_m': distance_m,
                            'distance_method': distance_method,
                            'distance_pinhole_m': distance_pinhole_m,
                            'distance_pitch_m': distance_pitch_m,
                            'bearing_deg': bearing_deg,
                            'cam_height_m': camera.height_m,
                            'cam_pitch_deg': camera.pitch_deg,  # Use actual telemetry
                            'cam_yaw_deg': camera.yaw_deg,
                            'cam_roll_deg': camera.roll_deg,
                            'cam_lat': camera.lat,
                            'cam_lon': camera.lon,
                            'person_geoposition': {
                                'latitude': person_lat,
                                'longitude': person_lon
                            } if person_lat is not None else None,
                        })
        return detections_info

    def find_geoposition_by_bbox(self, detections, bbox):
        """Find geoposition information for a detection by its bounding box."""
        if not detections or not bbox:
            return None
        
        for det in detections:
            if det.get('bbox') == list(bbox) or det.get('bbox') == bbox:
                return det.get('person_geoposition')
        
        return None
