import cv2
import os
import numpy as np
from pathlib import Path
from detection_pipeline import DetectionPipeline
from detection_fusion import DualDroneFusion, FrameSynchronizer, Detection
from camera import Camera


class DualDroneDetectionPipeline:
    """Pipeline for processing synchronized video streams from two drones."""
    
    def __init__(self, model_path: str, person_confidence_threshold: float = 0.7,
                 enable_weapon_detection: bool = True, 
                 weapon_confidence_threshold: float = 0.7,
                 sample_majority_threshold: int = 1,
                 association_threshold: float = 2.0):
        #association_threshold: Distance threshold for cross-drone detection matching (meters)
        
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
        
        # Initialize fusion module
        self.fusion = DualDroneFusion(association_threshold_m=association_threshold)
        
        # Initialize frame synchronizer
        self.synchronizer = FrameSynchronizer()
        
        # Initialize camera models for estimation (one per drone).
        # Using separate instances avoids overwriting pose/metadata when processing frame pairs.
        self.camera_drone1 = Camera()
        self.camera_drone2 = Camera()
        
        # Share statistics between pipelines (use drone1's stats as primary)
        self.stats = self.pipeline_drone1.stats
        self.pipeline_drone2.stats = self.stats
        
        # Settings
        self.save_crops = True
        self.enable_weapon_detection = enable_weapon_detection
        self.person_confidence_threshold = person_confidence_threshold

    def _select_best_detection(self, detections):
        if not detections:
            return None
        def score(d):
            try:
                return float(d.get('person_confidence', d.get('confidence', 0.0)))
            except Exception:
                return 0.0
        return max(detections, key=score)

    def _select_best_fused(self, fused_detections):
        if not fused_detections:
            return None
        fused_only = [d for d in fused_detections if d.get('source') == 'fused']
        return self._select_best_detection(fused_only) or self._select_best_detection(fused_detections)

    def _fmt_dist(self, value):
        try:
            if value is None:
                return "N/A"
            v = float(value)
            if not np.isfinite(v):
                return "N/A"
            return f"{v:.1f}m"
        except Exception:
            return "N/A"

    def _fmt_conf(self, value):
        try:
            if value is None:
                return "N/A"
            v = float(value)
            if not np.isfinite(v):
                return "N/A"
            return f"{v:.3f}"
        except Exception:
            return "N/A"

    def _fmt_geo(self, geo_dict):
        if not geo_dict:
            return "N/A"
        try:
            lat = float(geo_dict.get('latitude'))
            lon = float(geo_dict.get('longitude'))
            if not (np.isfinite(lat) and np.isfinite(lon)):
                return "N/A"
            return f"({lat:.6f},{lon:.6f})"
        except Exception:
            return "N/A"

    def _print_required_console_output(self, detections1, detections2, fused_detections):
        """Print ONLY the required two-line output for this frame."""
        best1 = self._select_best_detection(detections1)
        best2 = self._select_best_detection(detections2)
        best_fused = self._select_best_fused(fused_detections)

        d1_h = best1.get('distance_pinhole_m') if best1 else None
        d1_t = best1.get('distance_pitch_m') if best1 else None
        d2_h = best2.get('distance_pinhole_m') if best2 else None
        d2_t = best2.get('distance_pitch_m') if best2 else None

        geo1 = best1.get('person_geoposition') if best1 else None
        geo2 = best2.get('person_geoposition') if best2 else None

        geo_fused = best_fused.get('fused_geoposition') if best_fused else None
        d_fused = None
        if geo_fused and (getattr(self.camera_drone1, 'datetime', None) is not None) and (getattr(self.camera_drone2, 'datetime', None) is not None):
            try:
                from position_estimation import distance_from_position
                d_a = distance_from_position(self.camera_drone1, geo_fused['latitude'], geo_fused['longitude'])
                d_b = distance_from_position(self.camera_drone2, geo_fused['latitude'], geo_fused['longitude'])
                vals = [v for v in [d_a, d_b] if v is not None and np.isfinite(float(v))]
                if vals:
                    d_fused = float(np.mean(vals))
            except Exception:
                d_fused = None

        c1 = best1.get('person_confidence', best1.get('confidence')) if best1 else None
        c2 = best2.get('person_confidence', best2.get('confidence')) if best2 else None
        c_fused = best_fused.get('person_confidence', best_fused.get('confidence')) if best_fused else None

        print(
            "DISTANCE: "
            f"d1_height-based={self._fmt_dist(d1_h)}, "
            f"d1_pitch-based={self._fmt_dist(d1_t)}, "
            f"d2_height-based={self._fmt_dist(d2_h)}, "
            f"d2_pitch-based={self._fmt_dist(d2_t)}, "
            f"geo1={self._fmt_geo(geo1)}, "
            f"geo2={self._fmt_geo(geo2)}, "
            f"d_fused={self._fmt_dist(d_fused)}, "
            f"geo_fused={self._fmt_geo(geo_fused)}"
        )
        print(
            "DETECTIONS: "
            f"c1={self._fmt_conf(c1)}, "
            f"c2={self._fmt_conf(c2)}, "
            f"c_fused={self._fmt_conf(c_fused)}"
        )
        
    def process_dual_drone_samples(self, input_dir_drone1, input_dir_drone2, output_base_dir):
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
            return
        
        # Create output structure
        detections_base_dir = os.path.join(output_base_dir, "detections_dual_drone")
        crops_base_dir = os.path.join(output_base_dir, "crops_dual_drone")
        fused_base_dir = os.path.join(output_base_dir, "fused_detections")
        
        # Reset statistics for batch processing
        self.stats._in_batch_mode = True
        
        # Process each synchronized pair
        for sample_idx, sample_name in enumerate(sample_pairs, 1):
            _ = sample_idx  # no console logging
            
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
        
        # Finalize statistics
        self.stats.finalize()
        
    def _detect_people_with_estimation(self, image_path, image, drone_id):
        """Detect people and estimate their distance/bearing using estimation.py methods."""
        from position_estimation import estimate_distance, estimate_distance_2, estimate_bearing
        # Select the appropriate detector
        detector = self.pipeline_drone1.detector if drone_id == 1 else self.pipeline_drone2.detector
        camera = self.camera_drone1 if drone_id == 1 else self.camera_drone2
        # Run YOLO detection
        results = detector.model(image, imgsz=640, iou=0.6, conf=self.person_confidence_threshold, classes=[0], verbose=False)
        # Try to load metadata from JSON. If not available, we still return detections
        # (using filename-derived height/pitch when possible), so counts don't drop to 0.
        has_metadata = camera.load_from_json(image_path)
        height_annotated_m = None
        camera_pitch_annotated_deg = None
        try:
            file_meta = detector.extract_filename_metadata(image_path)
            height_annotated_m = file_meta.get('height_m')
            camera_pitch_annotated_deg = file_meta.get('camera_pitch_deg')
        except Exception:
            height_annotated_m = None
            camera_pitch_annotated_deg = None
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

                        # Always compute pinhole estimate.
                        distance_pinhole_m = estimate_distance(camera, person_height_px) if person_height_px > 0 else None

                        # Compute pitch-based estimate only when we have camera height + pitch.
                        distance_pitch_m = None
                        if has_metadata:
                            distance_pitch_m = estimate_distance_2(camera, y_bottom)

                        # Choose primary distance for downstream use.
                        if distance_pitch_m is not None:
                            distance_m = distance_pitch_m
                            distance_method = 'pitch'
                        elif distance_pinhole_m is not None:
                            distance_m = distance_pinhole_m
                            distance_method = 'pinhole'
                        else:
                            distance_m = None
                            distance_method = 'none'

                        bearing_deg = estimate_bearing(camera, x_center)

                        # For consistency with single-drone mode, also estimate a simple target geoposition.
                        person_lat, person_lon = None, None
                        if has_metadata and distance_m is not None and distance_m > 0:
                            try:
                                person_lat, person_lon = camera.calculate_person_geoposition(
                                    camera_lat=camera.lat,
                                    camera_lon=camera.lon,
                                    camera_yaw_deg=camera.yaw_deg,
                                    x_pixel=x_center,
                                    distance_m=distance_m
                                )
                            except Exception as e:
                                _ = e

                        detections_info.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'person_confidence': confidence,
                            'distance_m': distance_m,
                            'distance_method': distance_method,
                            'distance_pinhole_m': distance_pinhole_m,
                            'distance_pitch_m': distance_pitch_m,
                            'bearing_deg': bearing_deg,
                            'camera_height_annotated_m': height_annotated_m,
                            'camera_pitch_annotated_deg': camera_pitch_annotated_deg,
                            'camera_height_real_m': (camera.camera_height_m if has_metadata else None),
                            'camera_pitch_real_deg': (camera.pitch_deg if has_metadata else None),
                            'camera_yaw_real_deg': (camera.yaw_deg if has_metadata else None),
                            'metadata': (
                                {
                                    'gps': {
                                        'latitude': camera.lat,
                                        'longitude': camera.lon,
                                        'camera_height_real_m': camera.camera_height_m
                                    },
                                    'orientation': {
                                        'yaw': camera.yaw_deg,
                                        'pitch': camera.pitch_deg
                                    },
                                    'timestamp': camera.timestamp_sec,
                                    'datetime': camera.datetime
                                }
                                if has_metadata
                                else None
                            )
                        })

                        if person_lat is not None and person_lon is not None:
                            detections_info[-1]['person_geoposition'] = {
                                'latitude': person_lat,
                                'longitude': person_lon,
                                'bearing_deg': bearing_deg
                            }
        return detections_info

    def _add_fused_geopositions(self, fused_detections, detections1, detections2):
        """
        For matched detections (source='fused'), compute a fused target
        position from both drone camera poses + (distance,bearing) using least-squares.
        """
        from position_estimation import fuse_target_geoposition
        from camera import Camera

        if not fused_detections:
            return

        # Index original detections by bbox for quick lookup.
        det1_by_bbox = {tuple(d.get('bbox', [])): d for d in (detections1 or [])}
        det2_by_bbox = {tuple(d.get('bbox', [])): d for d in (detections2 or [])}

        for det in fused_detections:
            if det.get('source') != 'fused':
                continue

            bbox1 = det.get('bbox_drone1')
            bbox2 = det.get('bbox_drone2')
            d1 = det1_by_bbox.get(tuple(bbox1)) if bbox1 is not None else None
            d2 = det2_by_bbox.get(tuple(bbox2)) if bbox2 is not None else None
            if not d1 or not d2:
                continue

            dist1 = d1.get('distance_m')
            bear1 = d1.get('bearing_deg')
            dist2 = d2.get('distance_m')
            bear2 = d2.get('bearing_deg')
            if dist1 is None or bear1 is None or dist2 is None or bear2 is None:
                continue

            gps1 = (d1.get('metadata') or {}).get('gps') or {}
            gps2 = (d2.get('metadata') or {}).get('gps') or {}
            ori1 = (d1.get('metadata') or {}).get('orientation') or {}
            ori2 = (d2.get('metadata') or {}).get('orientation') or {}

            if 'latitude' not in gps1 or 'longitude' not in gps1 or 'latitude' not in gps2 or 'longitude' not in gps2:
                continue

            cam1 = Camera(lat=float(gps1['latitude']), lon=float(gps1['longitude']))
            cam2 = Camera(lat=float(gps2['latitude']), lon=float(gps2['longitude']))
            # Orientation is not strictly required for triangulation since bearings are already computed,
            # but we keep it for completeness/debugging.
            if 'yaw' in ori1:
                cam1.yaw_deg = float(ori1['yaw'])
            if 'pitch' in ori1:
                cam1.pitch_deg = float(ori1['pitch'])
            if 'yaw' in ori2:
                cam2.yaw_deg = float(ori2['yaw'])
            if 'pitch' in ori2:
                cam2.pitch_deg = float(ori2['pitch'])

            try:
                fused_lat, fused_lon = fuse_target_geoposition(cam1, float(dist1), float(bear1), cam2, float(dist2), float(bear2))
                det['fused_geoposition'] = {'latitude': fused_lat, 'longitude': fused_lon}
            except Exception as e:
                # Keep fusion results even if geoposition fails.
                _ = e
        
    def process_synchronized_sample_pair(self, sample_dir_drone1, sample_dir_drone2, output_base, sample_name, detections_base, crops_base, fused_base):
        # Get image files from both directories
        from config import SUPPORTED_FORMATS
        
        def get_image_files(directory):
            return sorted([
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.splitext(f)[1].lower() in [ext.lower() for ext in SUPPORTED_FORMATS]
            ])
        
        images_drone1 = get_image_files(sample_dir_drone1)
        images_drone2 = get_image_files(sample_dir_drone2)
        
        # No console logging
        
        # Synchronize frames by filename/index
        synchronized_pairs = self.synchronizer.synchronize_by_frame_index(
            images_drone1, images_drone2
        )
        
        # No console logging
        
        # Create output directories for this sample
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
    
    def process_frame_pair(self, frame1_path, frame2_path, frame_idx, output_det_d1, output_det_d2, output_fused, output_crops_d1, output_crops_d2, sample_name):
        # Load images
        img1 = cv2.imread(frame1_path)
        img2 = cv2.imread(frame2_path)
        
        # Get filenames
        filename1 = os.path.basename(frame1_path)
        filename2 = os.path.basename(frame2_path)
        name1, ext1 = os.path.splitext(filename1)
        name2, ext2 = os.path.splitext(filename2)
        
        # Detect people using current estimation methods
        detections1 = self._detect_people_with_estimation(frame1_path, img1, drone_id=1)
        detections2 = self._detect_people_with_estimation(frame2_path, img2, drone_id=2)

        # Testing-time condition metadata (used for RMSE comparisons only).
        frame_meta = self.pipeline_drone1.detector.extract_filename_metadata(frame1_path)
        real_distance = frame_meta.get('distance_m')
        camera_height_annotated = frame_meta.get('height_m')

        camera_pitch_annotated_deg = frame_meta.get('camera_pitch_deg')

        # Use real camera pitch from drone metadata (JSON) when available.
        camera_pitch_real_deg = None
        try:
            if self.camera_drone1.load_from_json(frame1_path):
                camera_pitch_real_deg = self.camera_drone1.pitch_deg
        except Exception:
            camera_pitch_real_deg = None
        
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
        fused_detections = self._fuse_frame_detections(
            detections1, detections2, weapon_results1, weapon_results2, frame_idx
        )

        # Required console output (ONLY).
        self._print_required_console_output(detections1, detections2, fused_detections)

        # Attach triangulated fused geoposition (no averaging).
        self._add_fused_geopositions(fused_detections, detections1, detections2)
        
        # Draw detections on individual frames
        img1_annotated = self._draw_boxes_with_fused_info(img1, detections1, weapon_results1, "Drone 1")
        img2_annotated = self._draw_boxes_with_fused_info(img2, detections2, weapon_results2, "Drone 2")
        
        # Save individual drone detections
        cv2.imwrite(os.path.join(output_det_d1, f"{name1}_detected{ext1}"), img1_annotated)
        cv2.imwrite(os.path.join(output_det_d2, f"{name2}_detected{ext2}"), img2_annotated)
        
        # Create visualization of fused detections (side-by-side view)
        fused_vis = self._create_fused_visualization(
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
            has_weapons_gt, distances1, pairs1, real_distance, camera_height_annotated,
            sample_class=sample_class,
            camera_pitch_annotated_deg=camera_pitch_annotated_deg,
            camera_pitch_real_deg=camera_pitch_real_deg,
            distance_pairs_pinhole=pairs1_p,
            distance_pairs_pitch=pairs1_pitch,
        )
        
        self.stats.add_image_results(
            len(detections2), weapons_detected_d2,
            len([r for r in weapon_results2 if r.get('has_weapons', False)]),
            has_weapons_gt, distances2, pairs2, real_distance, camera_height_annotated,
            sample_class=sample_class,
            camera_pitch_annotated_deg=camera_pitch_annotated_deg,
            camera_pitch_real_deg=camera_pitch_real_deg,
            distance_pairs_pinhole=pairs2_p,
            distance_pairs_pitch=pairs2_pitch,
        )
        
        # Fused detection stats (using fused confidence)
        self.stats.add_image_results(
            len(fused_detections), weapons_fused,
            weapons_fused,
            has_weapons_gt, [], [], None, None, sample_class, None
        )
        
          # No console logging
    
    def _draw_boxes_with_fused_info(self, image, detections, weapon_results, drone_label):
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
    
    def _fuse_frame_detections(self, detections1, detections2, weapon_results1, weapon_results2, frame_idx):
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
            
            # Get distance and bearing from detection
            distance = det.get('distance_m')  # Default if estimation failed
            bearing = det.get('bearing_deg')
            
            # Calculate ground-plane coordinates (relative to drone 1)
            # Using drone 1 position as (0, 0) reference point
            x = distance * np.sin(np.radians(bearing))
            y = distance * np.cos(np.radians(bearing))
            
            detection_objs1.append(Detection(
                bbox=tuple(det['bbox']),
                person_confidence=det['confidence'],
                distance_m=distance,
                bearing_deg=bearing,
                x=x,
                y=y,
                lat=0.0,
                lon=0.0,
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
            
            # Get distance and bearing from detection
            distance = det.get('distance_m')
            bearing = det.get('bearing_deg')
            
            # Calculate ground-plane coordinates
            # For drone 2, we need to offset relative to drone 1
            # Assuming drones are separated by ~10m for now (could be from GPS)
            drone2_offset_x = 10.0  # meters
            drone2_offset_y = 0.0
            
            x = drone2_offset_x + distance * np.sin(np.radians(bearing))
            y = drone2_offset_y + distance * np.cos(np.radians(bearing))
            
            detection_objs2.append(Detection(
                bbox=tuple(det['bbox']),
                person_confidence=det['confidence'],
                distance_m=distance,
                bearing_deg=bearing,
                x=x,
                y=y,
                lat=0.0,
                lon=0.0,
                drone_id=2,
                frame_id=frame_idx,
                has_weapon=has_weapon,
                weapon_confidence=weapon_conf
            ))
        
        # Perform fusion using fused confidence from DualDroneFusion
        fused = self.fusion.match_detections(detection_objs1, detection_objs2)
        return fused
    
    def _find_geoposition_by_bbox(self, detections, bbox):
        if not detections or bbox is None:
            return None
        for det in detections:
            if tuple(det.get('bbox', [])) == tuple(bbox):
                return det.get('person_geoposition')
        return None

    def _create_fused_visualization(self, img1, img2, fused_detections, detections1=None, detections2=None):
        # Create side-by-side layout
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

            geo1 = self._find_geoposition_by_bbox(detections1, bbox1)
            geo2 = self._find_geoposition_by_bbox(detections2, bbox2)

            def fmt_geo(g):
                if not g:
                    return "N/A"
                try:
                    return f"{float(g['latitude']):.6f},{float(g['longitude']):.6f}"
                except Exception:
                    return "N/A"

            geo_text = f"#{i+1}: D1={fmt_geo(geo1)} | D2={fmt_geo(geo2)}"

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
